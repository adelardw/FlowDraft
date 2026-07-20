import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from src.models.flowdraft_block_wise import FlowDraftBlockWise


class Orthrus(FlowDraftBlockWise):
    """Paper-faithful Orthrus masked-block baseline.

    A frozen AR pass builds the clean cache.  The diffusion pass then contains
    ``anchors_per_sequence`` independent blocks of size ``K`` (one visible
    anchor plus ``K - 1`` masks). Its
    4-D mask lets a block attend only to the AR prefix before its anchor and
    to itself bidirectionally; it can never read future clean tokens or any
    other block.  This is the dual-pass block-masking geometry of Orthrus,
    rather than full-sequence random token masking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orthrus.time_embed.requires_grad_(False)
        # Masked Orthrus consumes and trains the mask embedding that the
        # FlowDraft parent freezes as unused.
        self.orthrus.mask_embedding.requires_grad_(True)

    def _sample_anchors(self, attention_mask, block: int, count: int, document_ids=None):
        """Uniform anchor positions shared across the micro-batch.

        The paper uses a micro-batch of one; sharing positions also keeps a
        larger local batch compatible with one cache and one sparse mask.
        """
        min_prefix = int(self.cfg.train.get("min_prefix", 1))
        if attention_mask.size(1) < block:
            return None
        # ``block`` includes the visible anchor. A valid window contains the
        # anchor plus its K-1 drafted positions, all of which must be live.
        # With packed data, it must also be entirely within one source
        # document. In particular, the appended EOS can be predicted, but a
        # span beginning before EOS cannot continue into the next document.
        live_windows = attention_mask.bool().unfold(1, block, 1).all(dim=-1)
        valid = live_windows
        if document_ids is not None and bool(
            self.cfg.train.get("respect_document_boundaries", True)
        ):
            if document_ids.shape != attention_mask.shape:
                raise ValueError("document_ids must have the same shape as attention_mask")
            document_windows = document_ids.unfold(1, block, 1)
            valid = valid & (document_windows == document_windows[..., :1]).all(dim=-1)

        # Positions are shared across the local micro-batch because the DF
        # sparse mask and cache are batched. Keep only positions valid for
        # every row, then sample with replacement as the previous sampler did.
        shared = valid.all(dim=0)
        candidates = torch.nonzero(shared, as_tuple=False).flatten()
        candidates = candidates[candidates >= min_prefix]
        if candidates.numel() == 0:
            return None
        return candidates[torch.randint(candidates.numel(), (count,), device=attention_mask.device)]

    def _masked_step(self, batch):
        ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        block = int(self.cfg.train.get("block_size", 32))
        count = int(self.cfg.train.get("anchors_per_sequence", 256))
        anchors = self._sample_anchors(
            attention_mask, block, count, batch.get("document_ids")
        )
        if anchors is None:
            # Some packed batches contain no document long enough for one
            # complete block. Keep every trainable parameter connected so a
            # rank taking this path can still participate in DDP reduction.
            zero = sum(parameter.sum() for parameter in self.orthrus.df_parameters()) * 0.0
            drafted = max(block - 1, 0)
            empty_logits = ids.new_zeros(
                (ids.size(0), 0, drafted, 1), dtype=torch.float32
            )
            empty_live = torch.zeros(
                (ids.size(0), 0, drafted), dtype=torch.bool, device=ids.device
            )
            return zero, empty_logits, empty_logits, empty_live

        # One clean AR pass supplies both exact teacher rows and the shared
        # AR-only KV cache.  The DF forward below never appends to it.
        cache = DynamicCache(config=self.orthrus.model.config)
        with torch.no_grad(), self._teacher_eval():
            # The paper baseline packs every training/validation item to a
            # fixed 2048-token shape. This is the sole AR call routed through
            # the optional compiled wrapper; generation has variable cache
            # lengths and must remain eager to avoid recompilation churn.
            teacher_full = self.orthrus(
                ids, attention_mask, past_key_values=cache, use_compiled_ar=True
            ).logits

        width = block
        drafted = block - 1
        if drafted <= 0:
            raise ValueError("baseline block_size must be at least 2 (anchor + one mask)")
        anchor_ids = ids[:, anchors]  # [B, A]
        embed = self.orthrus.model.get_input_embeddings()
        anchor_embeds = embed(anchor_ids).unsqueeze(2)
        mask_embed = self.orthrus.mask_embedding.to(anchor_embeds.dtype)[None, None]
        masked_embeds = mask_embed.expand(ids.size(0), count, drafted, -1)
        x_in = torch.cat([anchor_embeds, masked_embeds], dim=2).flatten(1, 2)

        # Each synthetic block keeps the original absolute positions, even
        # though all blocks are concatenated for one efficient DF forward.
        offsets = torch.arange(width, device=ids.device)
        position_ids = (anchors[:, None] + offsets).flatten()[None].expand(ids.size(0), -1)
        # The official FlexAttention mask uses one causal limit per synthetic
        # query. It represents this sparse relation directly, instead of a
        # dense [8192, 10240] SDPA mask.
        # The clean anchor rides inside the diffusion block.  Its AR K/V must
        # therefore stay OUT of the historical cache view, exactly as at
        # inference where the pending anchor has not been committed yet.
        # ``_make_dual_pass_block_mask`` uses ``kv_idx <= causal_limit``, so
        # the final visible AR key is the token immediately before the anchor.
        causal_limit = (anchors - 1).repeat_interleave(width)[None].expand(
            ids.size(0), -1
        )
        df_all = self.orthrus(
            inputs_embeds=x_in,
            use_df=True,
            past_key_values=cache,
            position_ids=position_ids,
            causal_limit=causal_limit,
            diffusion_block_size=width,
        ).logits.view(ids.size(0), count, width, -1)
        # Causal-LM row j predicts the token after input row j.  Orthrus feeds
        # [anchor, mask, ..., mask], so the anchor through penultimate rows
        # predict the K-1 fresh tokens; the final mask row has no target in
        # this block.  This is also the convention used by official inference.
        df_logits = df_all[:, :, :-1]

        # AR row p predicts token p+1, exactly the first drafted position.
        gather = anchors[:, None] + torch.arange(drafted, device=ids.device)
        teacher_logits = teacher_full[:, gather]
        live = attention_mask[:, gather + 1].bool()
        loss = self._masked_kl_chunked(teacher_logits, df_logits, live)
        return loss, df_logits, teacher_logits, live

    def _masked_kl_chunked(self, teacher_logits, draft_logits, live):
        vocab = draft_logits.size(-1)
        teacher_flat = teacher_logits.reshape(-1, vocab)
        draft_flat = draft_logits.reshape(-1, vocab)
        live_flat = live.reshape(-1).float()
        n_live = live_flat.sum()

        def term(draft_chunk, teacher_chunk, live_chunk):
            log_draft = F.log_softmax(draft_chunk.float(), -1)
            log_teacher = F.log_softmax(teacher_chunk.float(), -1)
            kl = (log_teacher.exp() * (log_teacher - log_draft)).sum(-1)
            return (kl * live_chunk).sum()

        chunk = int(self.cfg.train.get("kl_chunk", 4096))
        total = draft_flat.new_zeros((), dtype=torch.float32)
        for start in range(0, draft_flat.size(0), chunk):
            stop = start + chunk
            total = total + checkpoint(
                term,
                draft_flat[start:stop],
                teacher_flat[start:stop],
                live_flat[start:stop],
                use_reentrant=False,
            )
        return total / n_live.clamp(min=1.0)

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._masked_step(batch)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {batch_idx}: {loss}")
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    @staticmethod
    def _position_metrics(df_logits, teacher_logits, live):
        """Greedy draft quality at every fresh-block position.

        ``accuracy_pos_j`` is independent teacher-top-1 agreement at position
        ``j``. ``acceptance_pos_j`` is stricter: it is one only when every
        greedy draft from position 1 through ``j`` matches its AR verifier
        row. The latter is exactly the event needed for a verifier to accept
        token ``j`` in a speculative block.
        """
        matches = df_logits.argmax(-1).eq(teacher_logits.argmax(-1)) & live
        # A padded position invalidates subsequent positions in that block;
        # valid packed baseline anchors have no padding, but keeping this
        # explicit makes validation with ordinary padded examples correct too.
        prefix_live = live.to(torch.int32).cumprod(dim=-1).bool()
        prefix_matches = matches.to(torch.int32).cumprod(dim=-1).bool()

        dims = (0, 1)  # aggregate local batch and sampled anchors
        accuracy_denominator = live.sum(dim=dims).clamp_min(1)
        acceptance_denominator = prefix_live.sum(dim=dims).clamp_min(1)
        accuracy = matches.sum(dim=dims) / accuracy_denominator
        acceptance = prefix_matches.sum(dim=dims) / acceptance_denominator

        full_blocks = prefix_live[..., -1]
        mean_accepted = (
            prefix_matches.sum(dim=-1)[full_blocks].float().mean()
            if full_blocks.any()
            else torch.zeros((), device=df_logits.device)
        )
        return accuracy, acceptance, mean_accepted

    def validation_step(self, batch, batch_idx):
        with self._frozen_val_rng(batch_idx):
            loss, df_logits, teacher_logits, live = self._masked_step(batch)
            live_count = live.sum().to(dtype=torch.float64)
            loss_stats = torch.stack(
                [loss.detach().to(torch.float64) * live_count, live_count]
            )
            loss_stats = self.trainer.strategy.reduce(loss_stats, reduce_op="sum")
            if loss_stats[1].item() > 0:
                # Already reduced across ranks. A skipped local batch carries
                # zero weight instead of looking like a perfect zero-loss batch.
                self.log(
                    "val/loss",
                    loss_stats[0] / loss_stats[1],
                    prog_bar=True,
                    sync_dist=False,
                )

                matches = df_logits.argmax(-1).eq(teacher_logits.argmax(-1)) & live
                prefix_live = live.to(torch.int32).cumprod(dim=-1).bool()
                prefix_matches = matches.to(torch.int32).cumprod(dim=-1).bool()
                dims = (0, 1)
                full_blocks = prefix_live[..., -1]
                metric_parts = [
                    matches.sum(dim=dims),
                    live.sum(dim=dims),
                    prefix_matches.sum(dim=dims),
                    prefix_live.sum(dim=dims),
                    prefix_matches.sum(dim=-1)[full_blocks].sum(),
                    full_blocks.sum(),
                ]
                metric_stats = torch.cat(
                    [part.reshape(-1).to(torch.float64) for part in metric_parts]
                )
                metric_stats = self.trainer.strategy.reduce(
                    metric_stats, reduce_op="sum"
                )
                drafted = live.size(-1)
                accuracy_num = metric_stats[:drafted]
                accuracy_den = metric_stats[drafted : 2 * drafted].clamp_min(1)
                acceptance_num = metric_stats[2 * drafted : 3 * drafted]
                acceptance_den = metric_stats[3 * drafted : 4 * drafted].clamp_min(1)
                accepted_sum, full_count = metric_stats[-2:]
                accuracy = accuracy_num / accuracy_den
                acceptance = acceptance_num / acceptance_den
                self.log_dict(
                    {
                        "val/teacher_agreement": accuracy_num.sum()
                        / metric_stats[drafted : 2 * drafted].sum().clamp_min(1),
                        **{
                            f"val/accuracy_pos_{position + 1:02d}": value
                            for position, value in enumerate(accuracy)
                        },
                        **{
                            f"val/acceptance_pos_{position + 1:02d}": value
                            for position, value in enumerate(acceptance)
                        },
                        "val/accepted_tokens_per_block": accepted_sum
                        / full_count.clamp_min(1),
                    },
                    sync_dist=False,
                )
            self._maybe_decode_val(batch, batch_idx)
        return loss

    def _draft_block(self, cache, block_size, times, sample: bool = False, anchor_token=None):
        if len(times) != 2:
            raise ValueError("the masked baseline drafts in exactly one step: use jumps=1")
        embed = self.orthrus.model.get_input_embeddings()
        # Orthrus defines K as the complete parallel block, including its
        # clean anchor.  Thus a K=32 baseline proposal contains 31 fresh
        # tokens.
        drafted = block_size - 1
        if drafted <= 0:
            raise ValueError("baseline block_size must be at least 2 (anchor + one mask)")
        masks = self.orthrus.mask_embedding.to(
            device=self.device, dtype=embed.weight.dtype
        ).expand(1, drafted, -1)
        if anchor_token is not None:
            anchor = embed(anchor_token.view(1, 1))
            x_in = torch.cat([anchor, masks], dim=1)
        else:
            x_in = masks
        mask = torch.ones(1, cache.get_seq_length() + x_in.size(1), dtype=torch.long, device=self.device)
        logits = self.orthrus(
            attention_mask=mask, inputs_embeds=x_in, use_df=True, past_key_values=cache
        ).logits
        # With a pending anchor, its output row predicts the first fresh token
        # and the final mask row is unused (the standard causal-LM shift).
        q = (logits[:, :-1] if anchor_token is not None else logits).float().softmax(-1)
        ids = torch.multinomial(q[0], 1).view(1, -1) if sample else q.argmax(-1)
        return ids, q
