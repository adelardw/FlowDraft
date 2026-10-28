import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from src.models.orthrus import Orthrus


class OrthrusOnPolicy(Orthrus):
    """Masked drafter with one on-policy term added to Orthrus's single-pass KL.

    ``train.onpolicy_term`` picks the term. The four settings form a 2x2 over
    two questions: does a pass read the drafter's own block, and is the label
    conditioned on that block.

    * ``none``     — no term; the single-pass KL with the base config's
      position weights.
    * ``opd``      — KL input: the single pass's own block. KL target:
      ``p_AR(. | ctx, a, d_<j)`` for the argmax block ``d`` of that same pass.
      Forward KL ``KL(p_AR || q)`` where the verifier accepts, reverse KL
      ``KL(q || p_AR)`` with weight ``gamma^(j-1)`` where it rejects, the two
      averaged — L_Draft-OPD, eq. (13) in "Draft-OPD: On-Policy Distillation
      for Speculative Draft Models" (Lei et al.).
    * ``chain_tf`` — KL input: refinement rounds over the drafter's own block,
      the most confident positions committed and the rest re-masked. KL
      target: ``p_AR(. | ctx, a, y_<j)``, teacher-forced on the training text.
    * ``chain``    — the same rounds. KL target: ``p_AR(. | ctx, a, d_<j)``,
      the verifier's response to the block the round reads.

    ``chain_tf`` and ``chain`` build the same states, run the same rounds and
    weigh positions by the same verdict; only the target differs. ``none`` and
    ``opd`` run a single pass. Decoding is Orthrus's, so a checkpoint of any
    setting decodes exactly like a masked drafter.
    """

    _TERMS = ("none", "opd", "chain_tf", "chain")

    def _onpolicy_term(self) -> str:
        term = str(self.cfg.train.get("onpolicy_term", "none"))
        if term not in self._TERMS:
            raise ValueError(f"unknown onpolicy_term='{term}' ({' | '.join(self._TERMS)})")
        return term

    def _masked_step(self, batch):
        """Orthrus's step, unchanged up to its single-pass KL, plus the term."""
        ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        block = int(self.cfg.train.get("block_size", 32))
        count = int(self.cfg.train.get("anchors_per_sequence", 256))
        anchors = self._sample_anchors(
            attention_mask, block, count, batch.get("document_ids")
        )
        if anchors is None:
            zero = sum(parameter.sum() for parameter in self.orthrus.df_parameters()) * 0.0
            drafted = max(block - 1, 0)
            empty_logits = ids.new_zeros(
                (ids.size(0), 0, drafted, 1), dtype=torch.float32
            )
            empty_live = torch.zeros(
                (ids.size(0), 0, drafted), dtype=torch.bool, device=ids.device
            )
            return zero, empty_logits, empty_logits, empty_live

        cache = DynamicCache(config=self.orthrus.model.config)
        with torch.no_grad(), self._teacher_eval():
            teacher_full = self.orthrus(
                ids, attention_mask, past_key_values=cache, use_compiled_ar=True
            ).logits

        width = block
        drafted = block - 1
        if drafted <= 0:
            raise ValueError("baseline block_size must be at least 2 (anchor + one mask)")
        anchor_ids = ids[:, anchors]
        embed = self.orthrus.model.get_input_embeddings()
        anchor_embeds = embed(anchor_ids).unsqueeze(2)
        mask_embed = self.orthrus.mask_embedding.to(
            device=anchor_embeds.device, dtype=anchor_embeds.dtype
        )[None, None]
        masked_embeds = mask_embed.expand(ids.size(0), count, drafted, -1)
        x_in = torch.cat([anchor_embeds, masked_embeds], dim=2).flatten(1, 2)

        offsets = torch.arange(width, device=ids.device)
        position_ids = (anchors[:, None] + offsets).flatten()[None].expand(ids.size(0), -1)
        causal_limit = (anchors - 1).repeat_interleave(width)[None].expand(
            ids.size(0), -1
        )
        df_kwargs = dict(
            position_ids=position_ids,
            causal_limit=causal_limit,
            diffusion_block_size=width,
        )
        df_all = self.orthrus(
            inputs_embeds=x_in, use_df=True, past_key_values=cache, **df_kwargs
        ).logits.view(ids.size(0), count, width, -1)
        df_logits = df_all[:, :, :-1]

        gather = anchors[:, None] + torch.arange(drafted, device=ids.device)
        teacher_logits = teacher_full[:, gather]
        live = attention_mask[:, gather + 1].bool()
        needs_w = (
            bool(self.cfg.train.get("acceptance_profile", None))
            or bool(self.cfg.train.get("position_weights", None))
            or float(self.cfg.train.get("teacher_chain_tail_weight", 1.0)) != 1.0
        )
        pos_w = self._position_weights(
            teacher_logits.flatten(1, 2),
            ids[:, gather + 1].reshape(ids.size(0), -1),
            live.flatten(1, 2),
        ) if needs_w else None
        loss = self._masked_kl_chunked(teacher_logits, df_logits, live, pos_w)

        term = self._onpolicy_term()
        if term == "opd":
            loss = loss + self._opd_term(
                df_logits, anchor_ids, attention_mask, cache, live, df_kwargs
            )
        elif term in ("chain", "chain_tf"):
            loss = loss + self._chain_term(
                df_logits, anchor_ids, attention_mask, cache, live, df_kwargs,
                embed, mask_embed,
                teacher_logits=teacher_logits if term == "chain_tf" else None,
            )
        return loss, df_logits, teacher_logits, live

    def _divergence_chunked(self, target_logits, draft_logits, weight, reverse):
        """``sum_rows weight * KL`` over full-vocabulary rows, chunked.

        reverse=False — ``KL(p || q)``, p from ``target_logits``, q from
        ``draft_logits``. reverse=True — ``KL(q || p)``. The target carries no
        gradient.
        """
        vocab = draft_logits.size(-1)
        p_flat = target_logits.reshape(-1, vocab)
        q_flat = draft_logits.reshape(-1, vocab)
        w_flat = weight.reshape(-1).float()

        def term(q_chunk, p_chunk, w_chunk):
            log_q = F.log_softmax(q_chunk.float(), -1)
            log_p = F.log_softmax(p_chunk.float(), -1)
            if reverse:
                kl = (log_q.exp() * (log_q - log_p)).sum(-1)
            else:
                kl = (log_p.exp() * (log_p - log_q)).sum(-1)
            return (kl * w_chunk).sum()

        chunk = self._kl_chunk_rows(vocab)
        total = q_flat.new_zeros((), dtype=torch.float32)
        for start in range(0, q_flat.size(0), chunk):
            stop = start + chunk
            total = total + checkpoint(
                term,
                q_flat[start:stop],
                p_flat[start:stop].detach(),
                w_flat[start:stop],
                use_reentrant=False,
            )
        return total

    def _opd_term(self, df_logits, anchor_ids, attention_mask, cache, live, df_kwargs):
        """The single pass, labelled by the verifier on its own block.

        Accepted positions ``j <= A``: ``KL(p_AR || q)``, averaged over them.
        Rejected positions ``j > A``: ``KL(q || p_AR)`` with weight
        ``gamma^(j-1)``, ``j`` counted from 1 in the block, normalised by the
        weights' sum. The term is the mean of the two parts, scaled by
        ``train.onpolicy_weight``.
        """
        weight = float(self.cfg.train.get("onpolicy_weight", 1.0))
        if weight <= 0.0 or not live.any():
            return df_logits.sum() * 0.0
        batch, count, drafted, _ = df_logits.shape
        flat_live = live.view(batch, -1)
        with torch.no_grad():
            draft_ids = df_logits.argmax(-1).view(batch, -1)
            target = self._verifier_response(
                draft_ids, anchor_ids, attention_mask, cache, df_kwargs
            )
            accepted, _ = self._greedy_verdict(
                draft_ids, target.argmax(-1), flat_live, count
            )
        rejected = flat_live & ~accepted
        gamma = float(self.cfg.train.get("opd_gamma", 0.8))
        decay = gamma ** torch.arange(drafted, device=df_logits.device, dtype=torch.float32)
        decay = decay.repeat(count)[None].expand(batch, -1)
        q = df_logits.flatten(1, 2)
        acc_w = accepted.float()
        rej_w = rejected.float() * decay
        acc = self._divergence_chunked(target, q, acc_w, reverse=False)
        rej = self._divergence_chunked(target, q, rej_w, reverse=True)
        acc = acc / acc_w.sum().clamp_min(1.0)
        rej = rej / rej_w.sum().clamp_min(1e-6)
        return weight * 0.5 * (acc + rej)

    def _chain_term(self, df_logits, anchor_ids, attention_mask, cache, live,
                    df_kwargs, embed, mask_embed, teacher_logits=None):
        """Refinement rounds over the drafter's own block.

        Round ``k`` reads the block with the ``round(31 (k+1) / (r+1))`` most
        confident positions committed to their argmax tokens and the rest
        masked — the state the decode schedule produces after pass ``k`` of
        ``r + 1``. The committed set only grows and its tokens stay fixed, as
        in decoding. KL target: the verifier's response to the block the round
        reads, or, with ``teacher_logits``, the teacher-forced target on the
        training text. Position weights are the same for both targets: full
        weight at the first position the verifier rejects,
        ``train.selfcorrect_tail_weight`` elsewhere, times ``dE/da_j``.
        """
        weight = float(self.cfg.train.get("selfcorrect_kl_weight", 0.0))
        if weight <= 0.0 or not live.any():
            return df_logits.sum() * 0.0
        batch, count, drafted, _ = df_logits.shape
        flat_live = live.view(batch, -1)
        rounds = max(1, int(self.cfg.train.get("selfcorrect_rounds", 2)))
        logits_cur = df_logits
        committed = torch.zeros(batch, count, drafted, dtype=torch.bool,
                                device=df_logits.device)
        tokens = torch.zeros(batch, count, drafted, dtype=torch.long,
                             device=df_logits.device)
        tf_target = None if teacher_logits is None else teacher_logits.flatten(1, 2)
        terms = []
        for k in range(rounds):
            with torch.no_grad():
                step_ids = logits_cur.argmax(-1)
                tokens = torch.where(committed, tokens, step_ids)
                draft_ids = tokens.view(batch, -1)
                response = self._verifier_response(
                    draft_ids, anchor_ids, attention_mask, cache, df_kwargs
                )
                accepted, known = self._greedy_verdict(
                    draft_ids, response.argmax(-1), flat_live, count
                )
                keep = self._commit_width(drafted, k, rounds + 1)
                confidence = logits_cur.float().softmax(-1).max(-1).values
                order = confidence.masked_fill(
                    committed, float("inf")).argsort(dim=-1, descending=True)
                committed = committed.clone()
                committed.scatter_(-1, order[..., :keep], True)
                slots = torch.where(
                    committed[..., None], embed(tokens),
                    mask_embed.expand(batch, count, drafted, -1),
                )
                x_in = torch.cat(
                    [embed(anchor_ids).unsqueeze(2), slots], dim=2
                ).flatten(1, 2)
            logits_next = self.orthrus(
                inputs_embeds=x_in, use_df=True, past_key_values=cache, **df_kwargs
            ).logits.view(batch, count, drafted + 1, -1)[:, :, :-1]
            tail = float(self.cfg.train.get("selfcorrect_tail_weight", 0.5))
            at_break = known & ~accepted
            pos_w = torch.where(
                at_break, torch.ones_like(confidence.view(batch, -1)),
                confidence.new_full((), tail),
            )
            surv = self._survival_weights(drafted, count, confidence)
            if surv is not None:
                pos_w = pos_w * surv
            target = response if tf_target is None else tf_target
            terms.append(self._teacher_loss(
                target, logits_next.flatten(1, 2), flat_live,
                position_weight=pos_w / pos_w.mean().clamp_min(1e-6),
            ))
            logits_cur = logits_next.detach()
        return weight * torch.stack(terms).mean()
