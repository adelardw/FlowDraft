import contextlib

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
    ``train.verify_kl_weight``, ``train.verify_s_uniform`` and
    ``train.min_prefix``.
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

    def _rollout_schedule(self, jumps, device):
        """Jump times ``0 = t_0 < ... < t_n = 1`` for one rollout.

        The interior boundaries are drawn one per equal bin of ``(0, 1)``
        rather than placed on a uniform grid, so successive steps see a
        different partition and the family is covered rather than sampled at
        the same n-1 points every time. Both ends are kept strictly inside
        ``(0, 1)``: the transport factor ``(t - s) / (1 - s)`` is unbounded as
        ``s -> 1``.
        """
        if jumps < 2:
            return [0.0, 1.0]
        interior = jumps - 1
        bins = torch.arange(interior, device=device, dtype=torch.float32)
        if bool(self.cfg.train.get("rollout_stratified", True)):
            offsets = torch.rand(interior, device=device)
        else:
            offsets = torch.full((interior,), 0.5, device=device)
        times = ((bins + offsets) / interior).clamp(1e-3, 1.0 - 1e-3)
        return [0.0] + times.sort().values.tolist() + [1.0]

    def _rollout_kl(self, teacher_logits, x1, block_mask, ctx_mask, cache,
                    anchor, df_kwargs):
        """Teacher alignment along the drafter's OWN multi-jump path.

        Every other term is evaluated on the interpolation path
        ``x_s = (1-s) x_0 + s x_1``, which is built from the answer. Decoding
        never sees that path: it starts from the prior and each step consumes
        the previous step's own output. Training on one and running on the
        other is the same exposure mismatch teacher forcing produces in an AR
        model, and nothing in the objective addresses it. This term rolls the
        drafter forward exactly as the decode loop does and supervises what it
        actually produces.

        It also gives the composition a direct signal. Endpoint consistency
        relates ``π_{s,t}`` to the diagonal pairwise; a schedule of ``n`` jumps
        executes a composition that no pairwise term ever evaluates end to end,
        so accumulated error is unconstrained.

        The target is the frozen AR distribution at every step, asked as "if
        you had to finish here, what would you emit" — that is ``π_{s_k,1}``,
        the map a schedule of any length terminates on. Supervising the
        intermediate horizons themselves would instead drive the family towards
        independence of ``t``, since the target does not depend on it.

        Gradients flow through the last ``train.rollout_grad_jumps`` steps
        only. The earlier ones run under ``no_grad``, so memory is bounded by
        that window rather than by the schedule length, and the recurrence is
        truncated before its conditioning degrades. Note the transport step is
        a convex combination with factor below one except at ``t = 1``, so the
        iteration is damped rather than expansive.
        """
        weight = float(self.cfg.train.get("rollout_kl_weight", 0.0))
        if weight <= 0.0:
            return None
        live = block_mask.bool()
        if not live.any():
            return None
        max_jumps = max(1, int(self.cfg.train.get("rollout_max_jumps", 8)))
        window = max(1, int(self.cfg.train.get("rollout_grad_jumps", 4)))
        anytime = bool(self.cfg.train.get("rollout_anytime", True))
        jumps = int(torch.randint(1, max_jumps + 1, (1,)))
        times = self._rollout_schedule(jumps, x1.device)
        pad = block_mask[..., None].to(x1.dtype)

        x = self.sample_prior(x1, block_mask)
        ones = x1.new_ones(x1.size(0))
        terms = []
        for step, (s_k, t_k) in enumerate(zip(times[:-1], times[1:])):
            grad_on = step >= jumps - window
            ctx = contextlib.nullcontext() if grad_on else torch.no_grad()
            s_vec = x1.new_full((x1.size(0),), s_k)
            x_in = x
            with ctx:
                logits = self._df_forward(
                    x_in, anchor, ctx_mask, cache,
                    s_vec, x1.new_full((x1.size(0),), t_k), df_kwargs,
                )
                pi = F.softmax(logits.float(), -1).to(x_in.dtype)
                # Endpoint prediction for this step's own horizon; the state
                # moves a (t-s)/(1-s) fraction of the way towards it.
                gamma = (t_k - s_k) / max(1.0 - s_k, 1e-3)
                x = (x_in + gamma * (pi - x_in)) * pad
            if not grad_on:
                continue
            if t_k >= 1.0:
                # The last step already asks the t = 1 question, so its own
                # output IS the "finish here" prediction — no extra forward.
                finish = logits
            elif anytime:
                # Same state, horizon moved to 1: what this step would emit if
                # the schedule stopped here.
                finish = self._df_forward(
                    x_in, anchor, ctx_mask, cache, s_vec, ones, df_kwargs,
                )
            else:
                continue
            terms.append(self._teacher_loss(teacher_logits, finish, live))
        if not terms:
            return None
        return torch.stack(terms).mean()

    def _sample_verify_s(self, like):
        """Noise levels for the last-jump verifier term.

        Sampled on a LOG scale, because a linear scale spends almost all of its
        budget on states that give the answer away. The interpolant is
        ``x_s = (1-s) x0 + s x1`` with ``x0 ~ Dirichlet(1)`` over the full
        vocabulary: at ``V ≈ 1.5e5`` the largest component of the prior is about
        ``ln(V)/V ≈ 9e-5``, so the clean token becomes the argmax of the input
        once ``s`` exceeds roughly ``1e-4``, and is linearly recoverable from
        the input embedding once ``s`` exceeds roughly ``V^{-1/2} ≈ 3e-3``. Above
        that the task degenerates into copying the input, and the drafter never
        sees such a state at decode time — a one-jump schedule enters at
        ``s = 0`` and a multi-jump schedule enters later steps on its own
        transported state, which is not an interpolation with the answer.

        Drawing ``log s`` uniformly on ``[log s_min, 0]`` puts about half the
        samples below ``1e-2``. ``s_min`` sits BELOW the argmax threshold on
        purpose: stopping at ``1e-4`` would leave every single sample in the
        regime where the clean token is the argmax of the input, which is the
        boundary of the leak rather than the far side of it. Stratification is applied
        in log space for the same reason as before: at a batch of one or two,
        i.i.d. draws leave coverage to chance.
        """
        batch = like.size(0)
        u = torch.rand_like(like)
        if bool(self.cfg.train.get("verify_s_stratified", True)):
            bins = torch.arange(batch, device=like.device, dtype=like.dtype)
            u = ((bins + u) / batch)[torch.randperm(batch, device=like.device)]
        if not bool(self.cfg.train.get("verify_s_log", True)):
            return u
        s_min = float(self.cfg.train.get("verify_s_min", 1e-6))
        log_min = torch.log(torch.tensor(s_min, device=like.device, dtype=like.dtype))
        return (log_min * (1.0 - u)).exp().clamp(0.0, 1.0 - 1e-3)

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
        prior = self.sample_prior(x1, block_mask)
        ones = torch.ones_like(t)
        if bool(self.cfg.train.get("verify_s_uniform", False)):
            # Align the verifier with the whole LAST-JUMP family. Any decode
            # schedule, however many jumps it takes, finishes on ``π_{s,1}``:
            # a one-jump run calls ``π_{0,1}(x_0)``, a two-jump run finishes on
            # ``π_{s,1}(x_s)`` at the intermediate noise level. Drawing
            # ``s ~ U[0,1)`` trains that family; ``s = 0`` is only its lower
            # endpoint, and the input there is pure noise carrying nothing
            # about ``x1``, so a fixed one-jump term supervises the least
            # informative point of it. Sampled independently of the ``(s, t)``
            # pair above so the two terms never share a trajectory.
            verify_s = self._sample_verify_s(s)
            verify_input = (
                (1.0 - verify_s[:, None, None]) * prior
                + verify_s[:, None, None] * x1
            )
        else:
            verify_s = torch.zeros_like(s)
            verify_input = prior
        verify_logits = self._df_forward(
            verify_input, anchor, ctx_mask, cache, verify_s, ones, df_kwargs
        )
        return (
            teacher_logits,
            draft_logits,
            verify_logits,
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
        #   KL input:  π^θ_{s,1}(x_s) — s = 0 by default (the pure prior, the
        #              exact one-jump map), or s ~ U[0,1) with
        #              train.verify_s_uniform, which covers the last jump of
        #              any multi-jump schedule.
        #   KL target: sg(p_AR) — the frozen AR path's distribution for the
        #              same block positions, already aligned in _shared_step.
        pos_w = self._position_weights(teacher_logits, x1, live)
        verify_kl = self._teacher_loss(
            teacher_logits, verify_logits, live, position_weight=pos_w
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
        # Teacher alignment on the diagonal. Combined with the endpoint CE this
        # is not a pair of competing targets: both are forward KLs, which are
        # linear in the target, so the weighted sum is minimised by the MIXTURE
        # (w_e * p(x1|x_t) + w_a * p_AR) / (w_e + w_a). What the flat weight
        # gets wrong is the blend ratio, which should not be constant in t —
        # see _ar_kl_sample_weight.
        ar_kl_weight = self.cfg.train.get("ar_kl_weight", 0.0)
        ar_kl = (
            self._teacher_loss(
                teacher_logits, diag_logits, live,
                sample_weight=self._ar_kl_sample_weight(t),
                position_weight=pos_w,
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

        rollout_kl_weight = float(self.cfg.train.get("rollout_kl_weight", 0.0))
        rollout_kl = self._rollout_kl(
            teacher_logits, x1, block_mask, ctx_mask, cache, anchor, df_kwargs
        )
        if rollout_kl is None:
            rollout_kl = draft_logits.sum() * 0.0

        lam = self._lambda()
        endpoint_weight = self.cfg.train.get(
            "endpoint_weight", self.cfg.train.get("anchor_weight", 1.0)
        )
        verify_kl_weight = self.cfg.train.get("verify_kl_weight", 0.0)
        loss = (
            verify_kl_weight * verify_kl
            + rollout_kl_weight * rollout_kl
            + endpoint_weight * endpoint
            + ar_kl_weight * ar_kl
            + lam * (4.0 * ec + 2.0 * td)
        )
        self.log_dict(
            {
                f"{metric_prefix}/endpoint": endpoint,
                f"{metric_prefix}/verify_kl": verify_kl,
                f"{metric_prefix}/rollout_kl": rollout_kl,
                f"{metric_prefix}/ar_kl": ar_kl,
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
            teacher_logits, draft_logits, verify_logits, *rest = self._shared_step(batch)
            loss = self.compute_loss(
                teacher_logits,
                draft_logits,
                verify_logits,
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
            self._log_prior_disagreement(rest)
            self._maybe_decode_val(batch, batch_idx)

    def _log_prior_disagreement(self, rest):
        """How much the drafter's endpoint prediction depends on the noise seed.

        Several priors are drawn for the SAME batch — same clean endpoints,
        same context, same noise level — and the spread of the resulting
        predictions is measured as the Jensen-Shannon divergence of the M
        predictive distributions, ``H(mean) - mean(H)``, averaged over live
        positions. Zero means the prediction is a function of the data alone;
        large values mean the seed is moving it.

        Two readings, at the two ends of the noise level:

        * ``s -> 0`` the input is independent of the endpoints, so the optimal
          prediction cannot depend on the seed and the divergence must fall to
          zero as training proceeds. If it plateaus above zero, the drafter is
          reading structure out of the noise.
        * ``s`` in the interior the input genuinely carries part of the answer,
          different seeds give different partial evidence, and a NON-zero
          spread is correct. A value near zero here means the opposite failure:
          the drafter ignores its input and predicts from context alone, which
          would make the whole diffusion path decorative.

        Diagnostic only, validation only, no gradient. Enable with
        ``train.prior_disagreement_samples > 1``.
        """
        samples = int(self.cfg.train.get("prior_disagreement_samples", 0))
        if samples < 2:
            return
        (_x_s, _x_t, x1, s, _t, ctx_mask, block_mask, cache, anchor,
         df_kwargs) = rest
        live = block_mask.bool()
        if not live.any():
            return
        ones = torch.ones_like(s)
        preds = []
        with torch.no_grad():
            for _ in range(samples):
                prior = self.sample_prior(x1, block_mask)
                x_seed = (1.0 - s[:, None, None]) * prior + s[:, None, None] * x1
                logits = self._df_forward(
                    x_seed, anchor, ctx_mask, cache, s, ones, df_kwargs
                )
                preds.append(F.softmax(logits.float(), -1))
        stack = torch.stack(preds)                       # [M, B, K, V]
        mean = stack.mean(0)
        # JSD = H(mean) - mean(H); both entropies in nats, so the value is
        # bounded by log(M) and comparable across runs with the same M.
        entropy = lambda p: -(p.clamp_min(1e-12).log() * p).sum(-1)
        jsd = entropy(mean) - entropy(stack).mean(0)
        self.log(
            "val/prior_disagreement",
            jsd[live].mean(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # Split by noise level: the two ends carry opposite expectations.
        low = s < 0.5
        for name, sel in (("low_s", low), ("high_s", ~low)):
            rows = sel[:, None] & live
            if rows.any():
                self.log(
                    f"val/prior_disagreement_{name}",
                    jsd[rows].mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
