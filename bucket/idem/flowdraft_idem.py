import torch

from src.models.flowdraft_block_wise import FlowDraftBlockWise


class FlowDraftIdem(FlowDraftBlockWise):
    """The block-wise flow map plus an idempotence term.

    Everything the parent trains is trained here unchanged: ``compute_loss``
    calls the parent first, with the same random draws the parent would see,
    and only then adds one term on top. Extra knob: ``train.idem_kl_weight``.

    Why the term exists. The refinement chain enters only at
    ``s ≥ selfcorrect_s_min``, so nothing supervises the map below that level.
    A decode schedule entering there lost more accepted tokens than a whole
    extra refinement pass buys. The term asks that a noised CORRECT answer be
    restored to itself at every level, the ones the chain never visits
    included.

    Under the discunif prior the clean token becomes the argmax of the input
    only past ``s = 1/2``. Above that the term is solved by copying and carries
    almost no gradient, so its signal lands exactly where the chain is blind.
    """

    def compute_loss(
        self,
        teacher_logits,
        verify_logits,
        x1,
        ctx_mask,
        block_mask,
        cache,
        anchor,
        anchor_ids,
        df_kwargs,
        onpolicy_logits=None,
        expected=None,
        accepted=None,
        known=None,
        *,
        metric_prefix="loss",
        log_on_step=True,
        log_on_epoch=False,
    ):
        loss = super().compute_loss(
            teacher_logits, verify_logits, x1,
            ctx_mask, block_mask, cache, anchor, anchor_ids, df_kwargs,
            onpolicy_logits=onpolicy_logits, expected=expected,
            accepted=accepted, known=known, metric_prefix=metric_prefix,
            log_on_step=log_on_step, log_on_epoch=log_on_epoch,
        )
        weight = float(self.cfg.train.get("idem_kl_weight", 0.0))
        live = block_mask.bool()
        # An empty block has already been logged as zeros by the parent, this
        # term's key included — see _log_zero_terms below.
        if weight <= 0.0 or not live.any():
            return loss

        # Idempotence on the correct answer.
        #   KL input:  π^θ_{s,1}(x_s), x_s = (1 − s)·ε + s·p, s ~ U[0, 1) per sample
        #   KL target: sg(p) — the target verify_kl is trained against
        exact = self._exact_target(teacher_logits, onpolicy_logits, known)
        pos_w = self._position_weights(teacher_logits, x1, live)
        p = exact.detach().float().softmax(-1).to(verify_logits.dtype)
        pad = block_mask[..., None].to(p.dtype)
        level = torch.rand(block_mask.size(0), device=block_mask.device)
        mix = level.to(p.dtype)[:, None, None]
        x_idem = ((1.0 - mix) * self.sample_prior(p, block_mask) + mix * p) * pad
        idem_logits = self._df_forward(
            x_idem, anchor, ctx_mask, cache, level, torch.ones_like(level), df_kwargs
        )
        idem_kl = self._teacher_loss(exact, idem_logits, live, position_weight=pos_w)
        self.log_dict(
            {f"{metric_prefix}/idem_kl": idem_kl},
            on_step=log_on_step, on_epoch=log_on_epoch, sync_dist=True,
        )
        return loss + weight * idem_kl

    def _log_zero_terms(self, metric_prefix, *, on_step, on_epoch):
        """The parent's zero keys, then this term's, in the order a full step
        logs them: collectives are matched by the order they are issued."""
        super()._log_zero_terms(metric_prefix, on_step=on_step, on_epoch=on_epoch)
        if float(self.cfg.train.get("idem_kl_weight", 0.0)) > 0.0:
            self.log_dict(
                {f"{metric_prefix}/idem_kl": torch.zeros((), device=self.device)},
                on_step=on_step, on_epoch=on_epoch, sync_dist=True,
            )
