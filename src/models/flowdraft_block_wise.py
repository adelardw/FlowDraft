import torch
import torch.nn.functional as F
from transformers import DynamicCache

from src.models.flowdraft import FlowDraft


class FlowDraftBlockWise(FlowDraft):
    """Block-wise training: the INFERENCE geometry reproduced at train time.

    The full-sequence variant (``FlowDraft``) noises the whole sequence and runs
    the DF path without a cache — a configuration the drafter never sees at
    decode time. Here every step looks exactly like one decode cycle:

      1. the batch is split at a random point ``p``; ONE no-grad AR forward
         over ``[:, :p+1+K]`` yields both the KV cache (cropped to ``p`` —
         the clean prefix, AR weights only) and the teacher distributions
         ``[p : p+K]`` for the K fresh tokens;
      2. the block mirrors the decode cycle exactly: position ``p`` is the
         CLEAN in-block anchor (the decode loop's pending correction/bonus
         token, whose K/V are not in the cache while drafting), positions
         ``p+1 .. p+K`` are noised and drafted;
      3. every DF forward of the loss (draft, anchor, EC expert, TD) runs
         WITH the cache AND the clean anchor at in-block position 0 — the
         drafter trains in the exact configuration it decodes in. All
         ``[B, T, V]`` tensors of the full-sequence variant shrink to ``[B, K, V]``.

    Same losses, samplers, knobs and checkpoints as the parent; extra knobs:
    ``train.block_size`` (K) and ``train.min_prefix``.
    """

    def _split_point(self, attention_mask, block: int, document_ids=None) -> int:
        """Sample the split from the TRUE lengths, not the padded width.

        With right-padding, sampling from ``ids.size(1)`` can drop the whole
        block window into pads (empty live mask -> NaN loss). The upper
        bound is the shortest live length in the batch; when even that is
        too short the split clamps to ``min_prefix`` and the per-position
        live mask (plus the empty-block guard in the loss) absorbs the rest.
        """
        min_prefix = int(self.cfg.train.get("min_prefix", 1))
        respect = bool(self.cfg.train.get("respect_document_boundaries", True))
        if document_ids is not None and respect:
            if document_ids.shape != attention_mask.shape:
                raise ValueError("document_ids must have the same shape as attention_mask")
            if attention_mask.size(1) < block + 1:
                raise ValueError("no packed document contains anchor + requested block")
            live_windows = attention_mask.bool().unfold(1, block + 1, 1).all(-1)
            doc_windows = document_ids.unfold(1, block + 1, 1)
            same_document = (doc_windows == doc_windows[..., :1]).all(-1)

            valid = live_windows & same_document
            candidates = torch.nonzero(valid.all(0), as_tuple=False).flatten()
            candidates = candidates[candidates >= min_prefix]
            if candidates.numel() == 0:
                raise ValueError(
                    "no packed document contains anchor + requested block"
                )
            choice = torch.randint(
                candidates.numel(), (1,), device=attention_mask.device
            )
            return int(candidates[choice])

        true_min = int(attention_mask.sum(dim=1).min())
        # the window is anchor + K fresh tokens: p + 1 + K must fit
        high = max(min_prefix + 1, true_min - block)
        return int(torch.randint(min_prefix, high, (1,)))

    def _df_forward(self, x_block, anchor, ctx_mask, cache, s, t):
        """One DF forward in the decode configuration: the clean anchor rides
        at in-block position 0, its output row is discarded — returned logits
        cover the K fresh positions only."""
        x_in = torch.cat([anchor, x_block], dim=1)
        logits = self.orthrus(x_in, ctx_mask, use_df=True, s=s, t=t, past_key_values=cache).logits
        return logits[:, 1:]

    def _prepare_block(self, batch):
        """The window math shared by the flow and baseline block variants:
        split at ``p``, ONE no-grad AR forward, clean-prefix cache, clean
        anchor, teacher for the K fresh tokens.

        With dynamic padding the tensor can be NARROWER than the window —
        the window shrinks to what exists, otherwise the teacher and block
        slices get clipped by the tensor edge to DIFFERENT lengths.
        """
        ids, mask = batch["input_ids"], batch["attention_mask"]
        document_ids = batch.get("document_ids")
        block = int(self.cfg.train.get("block_size", 64))
        p = self._split_point(mask, block, document_ids)
        width = ids.size(1)
        if width < 3:
            raise ValueError("cannot train on width<3 batches: prefix + anchor + block needed")
        p = min(p, width - 2)
        block = min(block, width - p - 1)
        # Match Orthrus packing semantics: the AR teacher and clean cache see
        # the complete causal packed prefix. Document IDs constrain only the
        # anchor + drafted window, so no proposal crosses an EOS boundary.
        # Keeping a common prefix length lets local batches larger than one
        # share a rectangular KV cache even when their document starts differ.
        ctx_mask = mask[:, : p + 1 + block]

        cache = DynamicCache(config=self.orthrus.model.config)
        with torch.no_grad(), self._teacher_eval():
            teacher_full = self.orthrus(
                ids[:, : p + 1 + block],
                ctx_mask,
                past_key_values=cache,
            ).logits
        # Keep ONLY the clean-prefix K/V: at decode time the pending anchor's
        # K/V are NOT in the cache while drafting. The AR logits at
        # [p : p+K] are the distributions of the fresh tokens p+1..p+K.
        cache.crop(p)
        teacher_logits = teacher_full[:, p : p + block]

        # position p — the clean in-block anchor; p+1..p+K — the drafted block
        anchor = self.df_processor.to_simplex(ids[:, p : p + 1], attention_mask=mask[:, p : p + 1])
        block_mask = mask[:, p + 1 : p + 1 + block]
        block_ids = ids[:, p + 1 : p + 1 + block]
        return teacher_logits, block_ids, ctx_mask, block_mask, cache, anchor

    def _shared_step(self, batch):
        teacher_logits, block_ids, ctx_mask, block_mask, cache, anchor = self._prepare_block(batch)
        x1 = self.df_processor.to_simplex(block_ids, attention_mask=block_mask)
        x_s, x_t, s, t = self.sample_trajectory(x1, block_mask)
        draft_logits = self._df_forward(x_s, anchor, ctx_mask, cache, s, t)
        return teacher_logits, draft_logits, x_s, x_t, x1, s, t, ctx_mask, block_mask, cache, anchor

    def compute_loss(self, teacher_logits, draft_logits, x_s, x_t, x1, s, t, ctx_mask, block_mask, cache, anchor):
        """The parent's three terms in block geometry.

        Differences from the full-sequence variant: no one-position shift (teacher
        is pre-aligned in ``_shared_step``), the live mask is the block's
        own, and every DF forward carries the clean-prefix cache AND the
        clean in-block anchor (via :meth:`_df_forward`).
        """
        eps = 1e-4
        live = block_mask.bool()
        if not live.any():
            # the whole block landed in padding: a zero step wired into the
            # graph instead of NaN from a mean over an empty tensor
            return draft_logits.sum() * 0.0
        log_draft = F.log_softmax(draft_logits.float(), -1)

        # Landing point of the jump — the EC-target input. Detached: the
        # jump's single teacher is ECLD.
        pi = log_draft.exp()
        gamma = ((t - s) / (1.0 - s).clamp(min=eps))[:, None, None]
        x_jump = x_s + gamma * (pi - x_s)
        x_jump = (x_jump * block_mask[..., None].to(x_jump.dtype)).detach()

        # --- categorical VFM endpoint likelihood on the diagonal. The
        # trajectory setting is paper-faithful; landing remains experimental.
        anchor_point = self.cfg.train.get("anchor_point", "trajectory")
        anchor_input = {"trajectory": x_t, "landing": x_jump}.get(anchor_point)
        if anchor_input is None:
            raise ValueError(f"unknown anchor_point='{anchor_point}' (trajectory | landing)")
        diag_logits = self._df_forward(anchor_input, anchor, ctx_mask, cache, s=t, t=t)
        endpoint_nll = F.cross_entropy(
            diag_logits.float().transpose(1, 2),
            x1.argmax(-1),
            reduction="none",
        )
        endpoint = endpoint_nll[live].mean()
        ar_kl_weight = self.cfg.train.get("ar_kl_weight", 0.0)
        ar_kl = (
            self._masked_kl(
                F.log_softmax(teacher_logits.float(), -1),
                F.log_softmax(diag_logits.float(), -1),
                live,
            )
            if ar_kl_weight
            else diag_logits.sum() * 0.0
        )

        # --- L_CE-EC — eq. (18) in "Categorical Flow Maps" (Roos et al.):
        # the jump must agree with the stop-grad expert at its landing point.
        if anchor_point == "landing":
            tgt = diag_logits.detach().float().softmax(-1)
        else:
            with torch.no_grad():
                tgt = self._df_forward(
                    x_jump, anchor, ctx_mask, cache, s=t, t=t
                ).float().softmax(-1)
        ec = -(tgt * log_draft).sum(-1)[live].mean()

        td = self._td_term(
            pi,
            gamma,
            s,
            t,
            live,
            forward_dt=lambda dt: self._df_forward(
                x_s, anchor, ctx_mask, cache, s=s, t=t + dt
            ),
        )

        lam = self._lambda()
        endpoint_weight = self.cfg.train.get(
            "endpoint_weight", self.cfg.train.get("anchor_weight", 1.0)
        )
        loss = (
            endpoint_weight * endpoint
            + ar_kl_weight * ar_kl
            + lam * (4.0 * ec + 2.0 * td)
        )
        self.log_dict(
            {
                "loss/endpoint": endpoint,
                "loss/ar_kl": ar_kl,
                "loss/ec": ec,
                "loss/td": td,
                "loss/lambda": lam,
            },
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(*self._shared_step(batch))
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {batch_idx}: {loss}")
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with self._frozen_val_rng(batch_idx):
            teacher_logits, draft_logits, *rest = self._shared_step(batch)
            loss = self.compute_loss(teacher_logits, draft_logits, *rest)
            block_mask = rest[-3]
            # Acceptance proxy in block geometry: no shift, teacher pre-aligned.
            agree = (draft_logits.argmax(-1) == teacher_logits.argmax(-1))[
                block_mask.bool()
            ]
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            self.log("val/teacher_agreement", agree.float().mean(), sync_dist=True)
            self._maybe_decode_val(batch, batch_idx)
        return loss
