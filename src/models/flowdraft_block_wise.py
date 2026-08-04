import torch
import torch.nn.functional as F
from transformers import DynamicCache

from src.models.flowdraft import FlowDraft


class FlowDraftBlockWise(FlowDraft):
    """Block-wise training: the INFERENCE geometry reproduced at train time.

    The full-sequence variant (``FlowDraft``) noises the whole sequence and runs
    the DF path without a cache — a configuration the drafter never sees at
    decode time. Here every step looks exactly like one decode cycle:

      1. one no-grad AR forward yields the clean cache and teacher
         distributions for one or more sampled blocks;
      2. the block mirrors the decode cycle exactly: position ``p`` is the
         CLEAN in-block anchor (the decode loop's pending correction/bonus
         token, whose K/V are not in the cache while drafting), positions
         ``p+1 .. p+K-1`` are noised and drafted;
      3. every DF forward of the loss (last-jump verifier alignment, draft,
         anchor, EC expert, TD) runs WITH the cache AND the clean anchor at
         in-block position 0. Multiple blocks are isolated with the same
         dual-pass mask used by Orthrus.

    Same losses, samplers, knobs and checkpoints as the parent; extra knobs:
    ``train.block_size`` (K = anchor + K-1 drafts), ``train.anchors_per_sequence``,
    ``train.verify_kl_weight``, ``train.teacher_chain_tail_weight``,
    ``train.position_weights`` and ``train.min_prefix``.
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
            if attention_mask.size(1) < block:
                raise ValueError("no packed document contains anchor + requested block")
            live_windows = attention_mask.bool().unfold(1, block, 1).all(-1)
            doc_windows = document_ids.unfold(1, block, 1)
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
        # The K-wide window is one anchor + K-1 fresh tokens.
        high = max(min_prefix + 1, true_min - block + 1)
        return int(torch.randint(min_prefix, high, (1,)))

    def _df_forward(
        self, x_block, anchor, ctx_mask, cache, s, t, df_kwargs=None
    ):
        """One DF forward in the decode configuration: the clean anchor rides
        at in-block position 0, its output row is discarded — returned logits
        cover the K-1 fresh positions only."""
        df_kwargs = df_kwargs or {}
        anchors = anchor.size(1)
        if anchors == 1:
            x_in = torch.cat([anchor, x_block], dim=1)
            logits = self.orthrus(
                x_in,
                ctx_mask,
                use_df=True,
                s=s,
                t=t,
                past_key_values=cache,
                **df_kwargs,
            ).logits
            return logits[:, 1:]

        if x_block.size(1) % anchors:
            raise ValueError("flattened drafted tokens must divide evenly across anchors")
        drafted = x_block.size(1) // anchors
        x_in = torch.cat(
            [
                anchor[:, :, None],
                x_block.view(x_block.size(0), anchors, drafted, -1),
            ],
            dim=2,
        ).flatten(1, 2)
        logits = self.orthrus(
            x_in,
            ctx_mask,
            use_df=True,
            s=s,
            t=t,
            past_key_values=cache,
            **df_kwargs,
        ).logits
        return logits.view(logits.size(0), anchors, drafted + 1, -1)[
            :, :, 1:
        ].flatten(1, 2)

    def _sample_anchor_points(
        self, attention_mask, block: int, count: int, document_ids=None
    ):
        """Sample shared valid anchors for one multi-block diffusion forward."""
        min_prefix = int(self.cfg.train.get("min_prefix", 1))
        if attention_mask.size(1) < block:
            return None
        valid = attention_mask.bool().unfold(1, block, 1).all(-1)
        if document_ids is not None and bool(
            self.cfg.train.get("respect_document_boundaries", True)
        ):
            if document_ids.shape != attention_mask.shape:
                raise ValueError("document_ids must have the same shape as attention_mask")
            windows = document_ids.unfold(1, block, 1)
            valid = valid & (windows == windows[..., :1]).all(-1)
        candidates = torch.nonzero(valid.all(0), as_tuple=False).flatten()
        candidates = candidates[candidates >= min_prefix]
        if candidates.numel() == 0:
            return None
        return candidates[
            torch.randint(candidates.numel(), (count,), device=attention_mask.device)
        ]

    def _prepare_block(self, batch):
        """The window math shared by the flow and baseline block variants:
        split at ``p``, ONE no-grad AR forward, clean-prefix cache, clean
        anchor, teacher for the K-1 fresh tokens.

        With dynamic padding the tensor can be NARROWER than the window —
        the window shrinks to what exists, otherwise the teacher and block
        slices get clipped by the tensor edge to DIFFERENT lengths.
        """
        ids, mask = batch["input_ids"], batch["attention_mask"]
        document_ids = batch.get("document_ids")
        block_width = int(self.cfg.train.get("block_size", 64))
        if block_width < 2:
            raise ValueError("block_size must be at least 2 (anchor + one draft)")
        p = self._split_point(mask, block_width, document_ids)
        width = ids.size(1)
        if width < 3:
            raise ValueError("cannot train on width<3 batches: prefix + anchor + draft needed")
        p = min(p, width - 2)
        block_width = min(block_width, width - p)
        drafted = block_width - 1
        # Match Orthrus packing semantics: the AR teacher and clean cache see
        # the complete causal packed prefix. Document IDs constrain only the
        # anchor + drafted window, so no proposal crosses an EOS boundary.
        # Keeping a common prefix length lets local batches larger than one
        # share a rectangular KV cache even when their document starts differ.
        ctx_mask = mask[:, : p + block_width]

        cache = DynamicCache(config=self.orthrus.model.config)
        with torch.no_grad(), self._teacher_eval():
            teacher_full = self.orthrus(
                ids[:, : p + block_width],
                ctx_mask,
                past_key_values=cache,
            ).logits
        # Keep ONLY the clean-prefix K/V: at decode time the pending anchor's
        # K/V are NOT in the cache while drafting. The AR logits at
        # [p : p+K-1] predicts the fresh tokens p+1..p+K-1.
        cache.crop(p)
        teacher_logits = teacher_full[:, p : p + drafted]

        # position p is the clean anchor; p+1..p+K-1 are drafted.
        anchor = self.df_processor.to_simplex(ids[:, p : p + 1], attention_mask=mask[:, p : p + 1])
        block_mask = mask[:, p + 1 : p + block_width]
        block_ids = ids[:, p + 1 : p + block_width]
        return teacher_logits, block_ids, ctx_mask, block_mask, cache, anchor

    def _position_weights(self, teacher_logits, x1, live):
        """Per-position weights for the teacher terms: chain consistency, then
        prefix survival.

        **Chain consistency.** Acceptance at position ``j`` is judged against
        ``p_AR`` conditioned on the tokens already accepted, and under greedy
        verification those are the frozen model's own greedy continuation. The
        single AR pass over the packed sequence conditions instead on the
        CORPUS tokens, and the two chains part company as soon as the model's
        top-1 disagrees with the data — after roughly two positions at the
        observed ~55% agreement. Up to that point the two conditionings are
        identical and the existing teacher is exactly right; past it the target
        is conditioned on a context the verifier is never in. So keep full
        weight up to the first disagreement and fall to
        ``train.teacher_chain_tail_weight`` after it. Costs nothing: the
        comparison uses logits that were already computed.

        Setting the tail to 1.0 restores the previous behaviour; 0.0 is a hard
        mask. A middle value trades bias for keeping the deep positions alive,
        which matters because they would otherwise receive gradient on only a
        few percent of blocks and could never improve ahead of the shallow ones.

        **Prefix survival.** The throughput metric is a conjunction over the
        prefix, ``E[len] = sum_j prod_{i<=j} a_i``, so the marginal value of
        position ``j`` is the probability of reaching it. Weighting by the
        measured survival curve moves gradient onto the early positions where
        the tokens actually are; uniform weighting spends most of it past
        position 4, which is reached on a few percent of blocks.
        """
        count = int(self.cfg.train.get("anchors_per_sequence", 1))
        drafted = teacher_logits.size(1) // count
        weights = None

        tail = self.cfg.train.get("teacher_chain_tail_weight", 1.0)
        if tail is not None and float(tail) != 1.0:
            matched = (
                teacher_logits.argmax(-1) == x1.argmax(-1)
            ).view(-1, count, drafted)
            keep = torch.ones_like(matched, dtype=teacher_logits.dtype)
            if drafted > 1:
                # position i survives iff every earlier position agreed
                keep[:, :, 1:] = matched[:, :, :-1].to(keep.dtype).cumprod(-1)
            weights = keep.flatten(1, 2) * (1.0 - float(tail)) + float(tail)

        survival = self.cfg.train.get("position_weights", None)
        if survival:
            w = torch.as_tensor(
                list(survival)[:drafted], device=teacher_logits.device,
                dtype=teacher_logits.dtype,
            )
            if w.numel() < drafted:
                w = torch.cat([w, w.new_full((drafted - w.numel(),), float(w[-1]))])
            w = (w / w.mean()).repeat(count)[None]
            weights = w if weights is None else weights * w
        if weights is None:
            return None
        return weights * live.to(weights.dtype)



    def _prepare_blocks(self, batch):
        """Prepare several isolated inference-geometry blocks in one DF pass.

        The frozen AR path runs once over the packed sequence. Synthetic
        K-wide ``[anchor + K-1 drafts]`` blocks are flattened along sequence length and the
        Orthrus dual-pass mask gives each one access only to the AR prefix
        preceding its own anchor and to itself bidirectionally.
        """
        ids, mask = batch["input_ids"], batch["attention_mask"]
        document_ids = batch.get("document_ids")
        block_width = int(self.cfg.train.get("block_size", 64))
        if block_width < 2:
            raise ValueError("block_size must be at least 2 (anchor + one draft)")
        drafted = block_width - 1
        count = int(self.cfg.train.get("anchors_per_sequence", 1))
        anchors = self._sample_anchor_points(
            mask, block_width, count, document_ids=document_ids
        )
        if anchors is None:
            raise ValueError("no packed document contains anchor + requested block")
        if self.orthrus.model.config.model_type != "qwen3":
            raise ValueError(
                "train.anchors_per_sequence > 1 currently requires the Qwen3 "
                "dual-pass attention path"
            )

        cache = DynamicCache(config=self.orthrus.model.config)
        with torch.no_grad(), self._teacher_eval():
            teacher_full = self.orthrus(
                ids, mask, past_key_values=cache
            ).logits

        offsets = torch.arange(drafted, device=ids.device)
        fresh_positions = anchors[:, None] + 1 + offsets
        teacher_positions = anchors[:, None] + offsets
        block_ids = ids[:, fresh_positions].flatten(1, 2)
        block_mask = mask[:, fresh_positions].flatten(1, 2)
        teacher_logits = teacher_full[:, teacher_positions].flatten(1, 2)
        anchor = self.df_processor.to_simplex(
            ids[:, anchors], attention_mask=mask[:, anchors]
        )

        position_ids = (
            anchors[:, None]
            + torch.arange(block_width, device=ids.device)[None]
        ).flatten()[None].expand(ids.size(0), -1)
        causal_limit = (anchors - 1).repeat_interleave(block_width)[None].expand(
            ids.size(0), -1
        )
        df_kwargs = {
            "position_ids": position_ids,
            "causal_limit": causal_limit,
            "diffusion_block_size": block_width,
        }
        return (
            teacher_logits,
            block_ids,
            mask,
            block_mask,
            cache,
            anchor,
            df_kwargs,
        )

    def _shared_step(self, batch):
        count = int(self.cfg.train.get("anchors_per_sequence", 1))
        if count > 1:
            prepared = self._prepare_blocks(batch)
        else:
            prepared = (*self._prepare_block(batch), {})
        (
            teacher_logits,
            block_ids,
            ctx_mask,
            block_mask,
            cache,
            anchor,
            df_kwargs,
        ) = prepared
        x1 = self.df_processor.to_simplex(block_ids, attention_mask=block_mask)
        x_s, x_t, s, t = self.sample_trajectory(x1, block_mask)
        draft_logits = self._df_forward(
            x_s, anchor, ctx_mask, cache, s, t, df_kwargs
        )
        # The decode loop enters every cycle at the pure prior, s = 0, so that
        # is the only state this term should be evaluated at.
        verify_input = self.sample_prior(x1, block_mask)
        verify_s = torch.zeros_like(s)
        ones = torch.ones_like(t)
        verify_logits = self._df_forward(
            verify_input, anchor, ctx_mask, cache, verify_s, ones, df_kwargs
        )

        # Warm-start term: the ONLY term that can teach the map to read its
        # input. At t = 1 the transport factor (t-s)/(1-s) is 1 for every s, so
        # the last jump returns π_{s,1}(x_s) and the incoming state contributes
        # nothing additively — all of a multi-jump schedule's value sits in
        # ∂π_{s,1}/∂x_s. Nothing else in the objective rewards that derivative:
        # verify_kl is evaluated at s = 0 on a fresh prior, and a map ignoring
        # x_s minimises it exactly as well.
        #
        # At decode the entry state need not be noise. The verify pass already
        # produces AR logits at every drafted position, so the next cycle can be
        # seeded with them at no extra forward. Those are a good-but-stale guess
        # — conditioned partly on tokens that were rejected. This term simulates
        # that state: mix the clean endpoint with the teacher's own prediction
        # shifted one position, at a level drawn from [warm_start_s_min, 1),
        # which sweeps "roughly right" through "exact".
        warm_weight = float(self.cfg.train.get("warm_start_weight", 0.0))
        if warm_weight:
            # The entry state is the teacher's OWN distribution read one
            # position out of place — informed, on the right manifold, and
            # wrong in the way a stale prediction is wrong. It must not be
            # mixed with x1: a convex mix with the clean endpoint puts the
            # answer at the argmax for any weight above a half, which recreates
            # the leak the prior change was made to remove.
            x_w = F.softmax(
                teacher_logits.roll(1, dims=1).float(), -1
            ).to(x1.dtype) * block_mask[..., None].to(x1.dtype)
            # s is a label, not a position on the interpolation path — this
            # state is off it. Train and decode only have to agree on the value.
            s_w = torch.full_like(s, float(self.cfg.train.get("warm_start_s", 0.5)))
            warm_logits = self._df_forward(
                x_w, anchor, ctx_mask, cache, s_w, ones, df_kwargs
            )
        else:
            warm_logits = None
        return (
            teacher_logits,
            draft_logits,
            verify_logits,
            warm_logits,
            x_s,
            x_t,
            x1,
            s,
            t,
            ctx_mask,
            block_mask,
            cache,
            anchor,
            df_kwargs,
        )

    def compute_loss(
        self,
        teacher_logits,
        draft_logits,
        verify_logits,
        warm_logits,
        x_s,
        x_t,
        x1,
        s,
        t,
        ctx_mask,
        block_mask,
        cache,
        anchor,
        df_kwargs,
        *,
        metric_prefix="loss",
        log_on_step=True,
        log_on_epoch=False,
    ):
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
        # Verifier alignment on the jump the decode loop finishes with.
        #   KL input:  π^θ_{0,1}(x_0) — the pure prior at s = 0, i.e.
        #              the state the decode loop enters every cycle at.
        #   KL target: sg(p_AR) — the frozen AR path's distribution for the
        #              same block positions, already aligned in _shared_step.
        pos_w = self._position_weights(teacher_logits, x1, live)
        verify_kl = self._teacher_loss(
            teacher_logits, verify_logits, live, position_weight=pos_w
        )
        warm_weight = float(self.cfg.train.get("warm_start_weight", 0.0))
        warm_kl = (
            self._teacher_loss(teacher_logits, warm_logits, live, position_weight=pos_w)
            if warm_logits is not None
            else draft_logits.sum() * 0.0
        )

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
        diag_logits = self._df_forward(
            anchor_input, anchor, ctx_mask, cache, s=t, t=t, df_kwargs=df_kwargs
        )
        endpoint_nll = F.cross_entropy(
            diag_logits.float().transpose(1, 2),
            x1.argmax(-1),
            reduction="none",
        )
        endpoint = endpoint_nll[live].mean()
        # --- L_CE-EC — eq. (18) in "Categorical Flow Maps" (Roos et al.):
        # the jump must agree with the stop-grad expert at its landing point.
        if anchor_point == "landing":
            tgt = diag_logits.detach().float().softmax(-1)
        else:
            with torch.no_grad():
                tgt = self._df_forward(
                    x_jump, anchor, ctx_mask, cache, s=t, t=t, df_kwargs=df_kwargs
                ).float().softmax(-1)
        ec = -(tgt * log_draft).sum(-1)[live].mean()

        td = self._td_term(
            pi,
            gamma,
            s,
            t,
            live,
            forward_dt=lambda dt: self._df_forward(
                x_s,
                anchor,
                ctx_mask,
                cache,
                s=s,
                t=t + dt,
                df_kwargs=df_kwargs,
            ),
        )

        lam = self._lambda()
        endpoint_weight = self.cfg.train.get(
            "endpoint_weight", self.cfg.train.get("anchor_weight", 1.0)
        )
        verify_kl_weight = self.cfg.train.get("verify_kl_weight", 0.0)
        loss = (
            verify_kl_weight * verify_kl
            + warm_weight * warm_kl
            + endpoint_weight * endpoint
            + lam * (4.0 * ec + 2.0 * td)
        )
        self.log_dict(
            {
                f"{metric_prefix}/endpoint": endpoint,
                f"{metric_prefix}/verify_kl": verify_kl,
                f"{metric_prefix}/warm_kl": warm_kl,
                f"{metric_prefix}/ec": ec,
                f"{metric_prefix}/td": td,
                f"{metric_prefix}/lambda": lam,
            },
            on_step=log_on_step,
            on_epoch=log_on_epoch,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(*self._shared_step(batch))
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {batch_idx}: {loss}")
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with self._frozen_val_rng(batch_idx):
            teacher_logits, draft_logits, verify_logits, warm_logits, *rest = self._shared_step(batch)
            loss = self.compute_loss(
                teacher_logits,
                draft_logits,
                verify_logits,
                warm_logits,
                *rest,
                metric_prefix="val/loss",
                log_on_step=False,
                log_on_epoch=True,
            )
            block_mask = rest[-4]
            # Exact one-jump training pair, rather than a random off-diagonal pair.
            agree = (verify_logits.argmax(-1) == teacher_logits.argmax(-1))[
                block_mask.bool()
            ]
            count = int(self.cfg.train.get("anchors_per_sequence", 1))
            block = verify_logits.size(1) // count
            matches = (
                verify_logits.argmax(-1) == teacher_logits.argmax(-1)
            ).view(verify_logits.size(0), count, block)
            live = block_mask.bool().view(block_mask.size(0), count, block)
            prefix_live = live.to(torch.int32).cumprod(-1).bool()
            prefix_matches = (matches & live).to(torch.int32).cumprod(-1).bool()
            acceptance = prefix_matches.sum((0, 1)) / prefix_live.sum(
                (0, 1)
            ).clamp_min(1)
            self.log(
                "val/loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log_dict(
                {
                    **{
                        # Same cross-model meaning as Orthrus: probability
                        # that the entire greedy prefix through this position
                        # agrees with the AR teacher in one drafter forward.
                        f"val/acceptance_pos_{position + 1:02d}": value
                        for position, value in enumerate(acceptance)
                    },
                },
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val/teacher_agreement",
                agree.float().mean(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self._maybe_decode_val(batch, batch_idx)
