import torch
import torch.nn.functional as F
from transformers import DynamicCache

from src.models.lit_orthrus import FlowMapOrthrus


class FlowMapOrthrusBlockWise(FlowMapOrthrus):
    """Block-wise training: the INFERENCE geometry reproduced at train time.

    The fixed variant (``FlowMapOrthrus``) noises the whole sequence and runs
    the DF path without a cache — a configuration the drafter never sees at
    decode time. Here every step looks exactly like one decode cycle:

      1. the batch is split at a random point ``p``; ONE no-grad AR forward
         over ``[:, :p+K]`` yields both the KV cache (cropped to ``p`` — the
         clean prefix, AR weights only) and the teacher distributions for
         the K block tokens (no extra shift needed afterwards: the slice
         ``[p-1 : p+K-1]`` is already token-aligned);
      2. only the K-token tail block is noised — every ``[B, T, V]`` tensor
         of the fixed variant shrinks to ``[B, K, V]`` (~T/K times smaller);
      3. all DF forwards (draft, anchor, EC expert, TD) run WITH the cache,
         attending to the clean prefix like at decode time.

    Same losses, samplers, knobs and checkpoints as the parent; extra knobs:
    ``train.block_size`` (K) and ``train.min_prefix``.
    """

    def _split_point(self, attention_mask, block: int) -> int:
        """Sample the split from the TRUE lengths, not the padded width.

        With right-padding, sampling from ``ids.size(1)`` can drop the whole
        block window into pads (empty live mask -> NaN loss). The upper
        bound is the shortest live length in the batch; when even that is
        too short the split clamps to ``min_prefix`` and the per-position
        live mask (plus the empty-block guard in the loss) absorbs the rest.
        """
        min_prefix = self.cfg.train.get("min_prefix", 1)
        true_min = int(attention_mask.sum(dim=1).min())
        high = max(min_prefix + 1, true_min - block + 1)
        return int(torch.randint(min_prefix, high, (1,)))

    def _shared_step(self, batch):
        ids, mask = batch["input_ids"], batch["attention_mask"]
        block = self.cfg.train.get("block_size", 64)
        p = self._split_point(mask, block)
        # With dynamic padding the tensor can be NARROWER than min_prefix + K.
        # Shrink the window to what actually exists — otherwise the teacher
        # slice [p-1 : p+K-1] and the block [p : p+K] get clipped by the
        # tensor edge to DIFFERENT lengths (off by one -> shape mismatch).
        width = ids.size(1)
        if width < 2:
            raise ValueError("cannot train on width-1 batches: no AR context for the block")
        p = min(p, width - 1)
        block = min(block, width - p)
        ctx_mask = mask[:, : p + block]

        cache = DynamicCache(config=self.orthrus.model.config)
        with torch.no_grad():
            teacher_full = self.orthrus(ids[:, : p + block], ctx_mask, past_key_values=cache).logits
        # Keep ONLY the clean-prefix K/V — the decode-loop invariant. The AR
        # logits at [p-1 : p+K-1] are the distributions of tokens p..p+K-1.
        cache.crop(p)
        teacher_logits = teacher_full[:, p - 1 : p + block - 1]

        block_mask = mask[:, p : p + block]
        x1 = self.df_processor.to_simplex(ids[:, p : p + block], attention_mask=block_mask)
        x_s, x_t, s, t = self.sample_trajectory(x1, block_mask)
        draft_logits = self.orthrus(
            x_s, ctx_mask, use_df=True, s=s, t=t, past_key_values=cache
        ).logits
        return teacher_logits, draft_logits, x_s, x_t, s, t, ctx_mask, block_mask, cache

    def compute_loss(self, teacher_logits, draft_logits, x_s, x_t, s, t, ctx_mask, block_mask, cache):
        """The parent's three terms in block geometry.

        Differences from the fixed variant: no one-position shift (teacher
        is pre-aligned in ``_shared_step``), the live mask is the block's
        own, and every DF forward carries the clean-prefix cache.
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

        # --- anchor: p_AR certifies the diagonal (see the parent for the
        # trajectory-vs-landing discussion; same knob).
        #   trajectory — KL input: π_{t,t}(x_t),          KL target: sg(p_AR(·|prefix, x1_{<i}))
        #   landing    — KL input: π_{t,t}(X_{s,t}(x_s)), KL target: sg(p_AR(·|prefix, x1_{<i}))
        anchor_point = self.cfg.train.get("anchor_point", "trajectory")
        anchor_input = {"trajectory": x_t, "landing": x_jump}.get(anchor_point)
        if anchor_input is None:
            raise ValueError(f"unknown anchor_point='{anchor_point}' (trajectory | landing)")
        diag_logits = self.orthrus(
            anchor_input, ctx_mask, use_df=True, s=t, t=t, past_key_values=cache
        ).logits
        anchor = self._masked_kl(
            F.log_softmax(teacher_logits.float(), -1),
            F.log_softmax(diag_logits.float(), -1),
            live,
        )

        # --- L_CE-EC — eq. (18) in "Categorical Flow Maps" (Roos et al.):
        # the jump must agree with the stop-grad expert at its landing point.
        if anchor_point == "landing":
            tgt = diag_logits.detach().float().softmax(-1)
        else:
            with torch.no_grad():
                tgt = self.orthrus(
                    x_jump, ctx_mask, use_df=True, s=t, t=t, past_key_values=cache
                ).logits.float().softmax(-1)
        ec = -(tgt * log_draft).sum(-1)[live].mean()

        # --- L_TD — eq. (16) in "Categorical Flow Maps" (Roos et al.):
        # ||∂_t π^θ_{s,t}||², finite difference in the time INPUT.
        dt = torch.where(t + 1e-3 <= 1.0, 1e-3, -1e-3)
        pi_dt = self.orthrus(
            x_s, ctx_mask, use_df=True, s=s, t=t + dt, past_key_values=cache
        ).logits.float().softmax(-1)
        drift = ((pi_dt - pi) / dt[:, None, None]).pow(2).sum(-1)
        td = (gamma.squeeze(-1) ** 2 * drift)[live].mean()

        loss = anchor + self.cfg.train.get("lambda", 1.0) * (4.0 * ec + 2.0 * td)
        self.log_dict({"loss/anchor": anchor, "loss/ec": ec, "loss/td": td})
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(*self._shared_step(batch))
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {batch_idx}: {loss}")
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        teacher_logits, draft_logits, *rest = self._shared_step(batch)
        loss = self.compute_loss(teacher_logits, draft_logits, *rest)
        block_mask = rest[-2]
        # Acceptance proxy in block geometry: no shift, teacher pre-aligned.
        agree = (draft_logits.argmax(-1) == teacher_logits.argmax(-1))[block_mask.bool()]
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/teacher_agreement", agree.float().mean())
        return loss
