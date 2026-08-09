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

    @staticmethod
    def _interleave_times(times, anchors, drafted):
        """``[B, A*drafted]`` per-position times -> the DF input's own layout.

        ``_df_forward`` splices a clean anchor row in front of every block, so a
        per-position time vector has to be spliced the same way. The anchor's
        slot gets ``1``: it is a token the verifier already committed, and at
        ``s = t = 1`` the transport factor is zero, so it stays exactly where it
        is while the drafted positions move.
        """
        batch = times.size(0)
        per_block = times.view(batch, anchors, drafted)
        head = times.new_ones(batch, anchors, 1)
        return torch.cat([head, per_block], dim=2).flatten(1, 2)

    def _df_forward(
        self, x_block, anchor, ctx_mask, cache, s, t, df_kwargs=None
    ):
        """One DF forward in the decode configuration: the clean anchor rides
        at in-block position 0, its output row is discarded — returned logits
        cover the K-1 fresh positions only.

        ``s``/``t`` are either one time per sequence (the usual case) or one per
        drafted position, shaped like ``x_block``; the second form is spliced
        into the anchor layout before it reaches the adapter.
        """
        df_kwargs = df_kwargs or {}
        anchors = anchor.size(1)
        if torch.is_tensor(s) and s.dim() == 2 and s.size(1) == x_block.size(1):
            drafted_per_anchor = x_block.size(1) // anchors
            s = self._interleave_times(s, anchors, drafted_per_anchor)
            t = self._interleave_times(t, anchors, drafted_per_anchor)
        if anchors == 1:
            x_in = torch.cat([anchor, x_block], dim=1)
            # Ширина блока сообщается адаптеру и на одноякорном пути. Без неё
            # он не знает, какие позиции — якоря, и гейт входа гасил бы якорь
            # вместе с приором. Раньше здесь шёл литеральный пустой df_kwargs,
            # поэтому исключение якоря не действовало НИ В ОДНОЙ руке стенда:
            # у всех anchors_per_sequence = 1.
            df_kwargs = {**df_kwargs,
                         "diffusion_block_size": x_in.size(1)}
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
        return (
            teacher_logits, block_ids, ctx_mask, block_mask, cache, anchor,
            ids[:, p : p + 1],
        )

    def _verifier_response(self, draft_ids, anchor_ids, ctx_mask, cache, df_kwargs):
        """ONE frozen AR forward over the drafter's OWN proposal.

        This is the forward the decode loop already spends on verification,
        moved into training, and it is the only source in the objective of two
        things that cannot be obtained any other way:

        * ``p_AR(· | ctx, draft_{<j})`` — a target CONDITIONED ON THE DRAFT.
          Every other teacher term in this file targets ``p_AR(· | corpus)``,
          which the drafter's state does not enter, so the state carries no
          gradient through the target. Here it does.
        * the EXACT greedy verdict. Acceptance is "the draft token equals what
          AR would emit given the draft tokens before it" — the conditioning is
          the draft's own prefix, not the corpus's. The corpus-conditioned
          teacher agrees with that only at the first drafted position; every
          deeper position was being judged against the wrong distribution.

        Returns logits aligned to the DRAFTED positions: entry ``j`` is the AR
        distribution of drafted position ``j`` given the anchor and drafted
        positions ``< j``. The trailing slot (AR's continuation past the block)
        is dropped, exactly as :meth:`verify_greedy` drops it.

        Two layouts, one meaning. With a single anchor the block sits directly
        behind a cache cropped to the split point, so an ordinary causal
        forward IS the block relation. With several isolated blocks flattened
        into one sequence the relation needs the dual-pass mask restricted to
        the past inside each block — ``weights='ar'`` runs the frozen backbone
        through exactly that geometry.
        """
        anchors = anchor_ids.size(1)
        drafted = draft_ids.size(1) // anchors
        ar_in = torch.cat(
            [anchor_ids[:, :, None], draft_ids.view(draft_ids.size(0), anchors, drafted)],
            dim=2,
        ).flatten(1, 2)
        committed = cache.get_seq_length()
        with torch.no_grad(), self._teacher_eval():
            if df_kwargs:
                logits = self.orthrus(
                    ar_in,
                    ctx_mask,
                    use_df=True,
                    weights="ar",
                    causal_in_block=True,
                    past_key_values=cache,
                    **df_kwargs,
                ).logits
            else:
                logits = self.orthrus(
                    ar_in, ctx_mask, past_key_values=cache
                ).logits
                # The AR path appends its K/V; the DF path crops for itself but
                # this call is not on it. Restore the committed prefix or every
                # later forward in the step reads a polluted cache.
                cache.crop(committed)
        width = drafted + 1
        return logits.view(logits.size(0), anchors, width, -1)[:, :, :-1].flatten(1, 2)

    def _exact_target(self, teacher_logits, onpolicy_logits, known):
        """Replace the corpus-conditioned target where the exact one is known.

        Greedy acceptance at position j is judged against ``p_AR`` conditioned
        on the tokens already accepted -- which, under greedy consensus, are the
        frozen model's own greedy continuation. The single AR pass over the
        packed sequence conditions on the CORPUS tokens instead, and the two
        agree only while the model's top-1 matches the data: about two positions
        at the measured ~55% per-position agreement. Every deeper position is
        therefore trained against a distribution the verifier is never in.
        ``teacher_chain_tail_weight`` currently models that with an indicator
        and a guessed discount.

        The on-policy sweep gives the exact thing for free. Up to the break the
        draft IS the AR argmax, so ``p_AR(· | ctx, draft_{<j})`` is precisely
        ``p_AR(· | ctx, greedy chain_{<j})``. Past the break the draft diverges
        and the sweep is conditioned on a rejected token, so the corpus target
        (with its discount) remains the better of two flawed options.

        This is a defect the masked baseline shares, so fixing it is not a
        correction -- it is an advantage, and it costs the one AR forward the
        self-correction terms already pay for. At initialisation the break sits
        at position zero and almost everything falls back to the corpus; the
        exact region grows as the drafter improves, which is the curriculum one
        would have designed by hand.
        """
        source = str(self.cfg.train.get("teacher_source", "corpus"))
        if source not in ("corpus", "onpolicy"):
            raise ValueError(f"unknown teacher_source='{source}' (corpus | onpolicy)")
        if source == "corpus" or onpolicy_logits is None or known is None:
            return teacher_logits
        return torch.where(known[..., None], onpolicy_logits, teacher_logits)

    @staticmethod
    def _greedy_verdict(draft_ids, expected, live, anchors):
        """What greedy verification does with this draft. Costs no forward.

        Returns ``(accepted, known)``: the accepted-prefix indicator and the
        set of positions the next cycle receives as settled — the accepted
        prefix plus the ONE position the verifier overwrites with its own
        token. Everything past that break is still open, and it was drafted
        under a prefix that has just been shown to be wrong.
        """
        agree = (draft_ids == expected) & live
        per_block = agree.view(agree.size(0), anchors, -1).to(torch.int32)
        accepted = per_block.cumprod(dim=2).bool()
        correction = (~accepted).cumsum(dim=2) == 1
        shape = draft_ids.shape
        return accepted.view(shape) & live, (accepted | correction).view(shape) & live

    def _entry_times(self, rounds, device):
        """Entry times ``s_1 < ... < s_r``, one per equal bin of ``(s_min, 1)``.

        The schedule a multi-jump decode executes is a sequence of RESTARTS,
        ``[(0,1), (s_1,1), ..., (s_{r-1},1)]``: every leg asks the terminal
        question from wherever the previous one left the draft. Stratifying
        rather than using a fixed grid covers the family instead of sampling the
        same r-1 points every step.

        ``train.selfcorrect_s_min`` decides how much of the draft the leg
        actually receives, and it is the load-bearing knob. The entry state is
        ``(1-s) x_0' + s q``, so at ``s = 0.25`` three quarters of the input is
        prior noise and the draft arrives at a quarter amplitude -- through a
        FROZEN embedding, with no input projection to rescale it. Pushed towards
        1 the input becomes the draft itself, and the leg is asked the
        well-posed question "read these tokens, take one AR step at every
        position in parallel", which is the Jacobi operator the prefix-fixing
        induction is stated for. Transport is unaffected either way: at ``t = 1``
        the factor ``(t-s)/(1-s)`` is 1 for every ``s``.
        """
        low = float(self.cfg.train.get("selfcorrect_s_min", 0.0))
        if not 0.0 <= low < 1.0:
            raise ValueError(
                f"train.selfcorrect_s_min must lie in [0, 1), got {low}"
            )
        bins = torch.arange(rounds, device=device, dtype=torch.float32)
        offsets = torch.rand(rounds, device=device)
        spread = (bins + offsets) / rounds
        return (low + (1.0 - low) * spread).clamp(1e-3, 1.0 - 1e-3).sort().values

    def _selfcorrect_kl(self, verify_logits, onpolicy_logits, accepted, known, anchor_ids,
                        block_mask, ctx_mask, cache, anchor, df_kwargs):
        """The multi-step term: the drafter walks its OWN jump schedule, and
        every leg is supervised by the verifier's answer to the leg before it.

        A schedule of ``n`` jumps executes a composition that no pairwise term
        evaluates end to end. Training one extra pair -- say ``(0.5, 1)`` -- buys
        two-step decoding at one point and leaves ``n = 3, 4`` exactly as
        unsupervised as before. So this walks the schedule:

            q_0 = pi_{0,1}(x_0)                            the deployed first jump
            for k = 1..r:
                target_k = p_AR(· | ctx, argmax q_{k-1})   ONE frozen AR sweep
                q_k      = pi_{s_k, 1}((1-s_k) x_0' + s_k q_{k-1})
                loss    += l(target_k, q_k)

        with ``s_1 < ... < s_r`` stratified in ``(0,1)``, so the entry times a
        real schedule visits are covered rather than fixed. Round 0's sweep is
        already in hand -- it is the same forward the verdict comes from -- so
        ``r`` rounds cost ``r`` DF forwards and ``r-1`` extra AR forwards.

        The fixed point of this recursion is the frozen model's greedy chain:
        ``q`` is fully accepted iff ``q_j = argmax p_AR(· | ctx, q_{<j})`` for
        every j. Each round is one Jacobi sweep towards it, so if the map
        realises the sweep, ``n`` composed jumps fix positions ``1..n``:
        ``A_n >= min(n, K-1)``, with a deterministic verifier and no new
        information. What makes reading the input pay is that the input CONTAINS
        a partial computation of the target -- the target is a sweep applied to
        the draft, and the draft is the input. Every corpus-conditioned term has
        an input that is empty in that sense, which is why none of them can
        reward the derivative a second jump lives on.

        States are detached between rounds. Backpropagating through the
        composition would let an early leg optimise a later leg's input, a
        self-referential objective on a map whose failure mode is collapse; the
        induction above needs each leg to sweep correctly, not the chain to be
        differentiated.

        The verdict weights, it does not mask, and the weighted set is ``known``
        -- the accepted prefix PLUS the break. Weighting by ``accepted`` alone
        gets it exactly backwards: the break is the one position whose target is
        both valid (its prefix was confirmed) and informative (the draft is
        wrong there), while the accepted prefix is where the draft is already
        right and carries little gradient. Past the break the target is
        conditioned on a token that has just been rejected, and
        ``train.selfcorrect_tail_weight`` says how far to trust it.
        """
        weight = float(self.cfg.train.get("selfcorrect_kl_weight", 0.0))
        if weight <= 0.0:
            return None
        live = block_mask.bool()
        if not live.any():
            return None
        rounds = max(1, int(self.cfg.train.get("selfcorrect_rounds", 2)))
        times = self._entry_times(rounds, block_mask.device).tolist()
        tail = float(self.cfg.train.get("selfcorrect_tail_weight", 0.5))
        pad = block_mask[..., None].to(verify_logits.dtype)
        ones = block_mask.new_ones(block_mask.size(0), dtype=torch.float32)

        q = verify_logits.detach().float().softmax(-1).to(verify_logits.dtype)
        target, verdict, accepted_k = onpolicy_logits, known, accepted
        terms = []
        for k, s_k in enumerate(times):
            prior = self.sample_prior(q, block_mask)
            x_in = ((1.0 - s_k) * prior + s_k * q) * pad
            logits = self._df_forward(
                x_in, anchor, ctx_mask, cache, ones * s_k, ones, df_kwargs
            )
            # Полный вес — на позицию РАЗРЫВА, а не на весь принятый префикс.
            # На принятых позициях таргет совпадает с токеном черновика, который
            # при s > 0.5 доминирует в собственном слоте входа: копирование даёт
            # там нулевой лосс, учиться нечему, а вес был максимальным. Разрыв —
            # единственное место, где таргет отличается от входа.
            pos_w = torch.where(
                verdict & ~accepted_k, torch.ones_like(pad[..., 0]),
                pad.new_full((), tail)
            )
            # Нормируется ПРОИЗВЕДЕНИЕ, и так же в маскирующей ветви.
            #
            # На сам лосс порядок не влияет вовсе: `_teacher_loss` делит на
            # реализованную массу веса, то есть инвариантен к масштабу
            # `position_weight`, а два порядка отличаются постоянным множителем
            # (замер: лосс 1.0273889 против 1.0273888, расхождение 1.2e-7 —
            # шум float32). Выбран тот, у которого средний вес РОВНО 1.000 при
            # любой приёмке: тогда логируемое число сравнимо между руками и по
            # ходу обучения, а у обратного порядка оно плывёт (1.057).
            # The prefix-conjunction weight is a property of the metric, not of
            # any one target, so it multiplies here exactly as it does in
            # verify_kl. The chain-validity gate does NOT: this term's target is
            # conditioned on the drafter's own block, so corpus agreement says
            # nothing about it.
            survival = self._survival_weights(
                block_mask.size(1) // anchor.size(1), anchor.size(1), pad
            )
            if survival is not None:
                pos_w = pos_w * survival
            pos_w = pos_w / pos_w.mean().clamp_min(1e-6)
            terms.append(
                self._teacher_loss(target, logits, live, position_weight=pos_w)
            )
            if k + 1 < len(times):
                q = logits.detach().float().softmax(-1).to(verify_logits.dtype)
                draft_ids = q.argmax(-1)
                target = self._verifier_response(
                    draft_ids, anchor_ids, ctx_mask, cache, df_kwargs
                )
                accepted_k, verdict = self._greedy_verdict(
                    draft_ids, target.argmax(-1), live, anchor.size(1)
                )
        return torch.stack(terms).mean()

    def _position_weights(self, teacher_logits, x1, live):
        # `x1` принимается и точкой симплекса [.., V], и прямо идентификаторами
        # [..]: внутри от неё нужен ровно argmax. Маскирующая ветвь строила
        # one-hot единственно ради этого argmax, а при бумажном пресете
        # (anchors=256, block=64) он весит ~3.2 ГБ на шаг.
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
            target_ids = x1 if x1.dim() == teacher_logits.dim() - 1 else x1.argmax(-1)
            matched = (
                teacher_logits.argmax(-1) == target_ids
            ).view(-1, count, drafted)
            keep = torch.ones_like(matched, dtype=teacher_logits.dtype)
            if drafted > 1:
                # position i survives iff every earlier position agreed
                keep[:, :, 1:] = matched[:, :, :-1].to(keep.dtype).cumprod(-1)
            gate = keep * (1.0 - float(tail)) + float(tail)
            # Normalise the gate PER POSITION, not globally. It selects which
            # samples carry a valid target; it is not a statement about how
            # much position j is worth. Left un-normalised it decays
            # geometrically at roughly the divergence hazard, and the survival
            # weights decay at nearly the same rate from an unrelated cause, so
            # the product applies one schedule twice — under-weighting position
            # 4 by about six times and position 7 by seventeen against the exact
            # metric gradient, which are the positions a second jump has to earn
            # its cost on. Per-position normalisation makes it a mean-one
            # validity gate and leaves the across-position schedule to u_j alone.
            # Нормировка по СКОЛЬЗЯЩЕМУ среднему, а не по текущему батчу.
            # Знаменатель здесь — оценка средней массы гейта на позиции, и
            # E[1/m] != 1/E[m]: на стенде батч даёт всего batch_size*anchors = 2
            # наблюдения, и смещение достигает −40% на позиции 4 и −51% на
            # позиции 7, причём с вероятностью 0.945 в батче нет ни одного
            # валидного сэмпла и гейт молча становится тождественной единицей.
            # То есть попозиционная нормировка вдвое резала ровно те глубокие
            # позиции, которые бралась защищать. Скользящее среднее делает
            # знаменатель оценкой по тысячам наблюдений. Оно не сохраняется в
            # чекпоинт и прогревается заново примерно за сотню шагов после
            # возобновления.
            batch_mean = gate.mean(dim=(0, 1), keepdim=True).detach()
            # Среднее хранится ПО ШИРИНЕ БЛОКА. Ширина батча динамическая
            # (pack_sequences=false), и на коротких строках drafted < K-1;
            # единственный буфер тогда сбрасывался на каждом таком шаге в
            # оценку по batch*anchors = 2 наблюдениям -- ровно то смещение до
            # -51%, ради устранения которого правка и делалась.
            store = getattr(self, "_gate_mass", None)
            if store is None:
                store = {}
                self._gate_mass = store
            key = int(batch_mean.shape[-1])
            prev = store.get(key)
            if self.training:
                prev = (batch_mean.clone() if prev is None
                        else 0.99 * prev.to(batch_mean) + 0.01 * batch_mean)
                store[key] = prev
            norm = prev if prev is not None else batch_mean
            gate = gate / norm.to(gate).clamp_min(1e-6)
            weights = gate.flatten(1, 2)

        w = self._survival_weights(drafted, count, teacher_logits)
        if w is not None:
            weights = w if weights is None else weights * w
        if weights is None:
            return None
        return weights * live.to(weights.dtype)

    def _survival_weights(self, drafted, count, like):
        """``dE/da_j`` alone, without the chain-validity gate.

        Acceptance is a conjunction over the prefix, ``E[len] = sum_j
        prod_{i<=j} a_i``, so position ``j`` is worth the probability of
        reaching it at all. That is a property of the METRIC and applies to
        every term that targets acceptance -- not only to the one whose target
        happens to be corpus-conditioned. The chain gate is separate: it says
        whether a corpus-conditioned target is valid at all, which is
        meaningless for a target conditioned on the drafter's own block.

        Предпочтительный вход — `train.acceptance_profile`, то есть ИЗМЕРЕННЫЕ
        попозиционные приёмки `a_j`, из которых вес считается точно:
        `dE/da_j = S_{j-1}(1 + R_j)`, `R_j = a_{j+1}(1 + R_{j+1})`, `R_n = 0`.
        Жёсткий список весов был нормирован на `S_1 = 1`, что молча
        предполагает `a_1 = 1`: `dE/da_1 = 1 + R_1` вообще не содержит `a_1`,
        тогда как остальные ему пропорциональны, поэтому самая ценная позиция
        занижалась ровно в `1/a_1` раз (при измеренном `a_1 = 0.525` — в 1.47).
        Вывод из приёмки заодно распространяется на любой размер блока:
        хвост продолжается последней приёмкой, а не последним ВЕСОМ, и
        выживание там падает геометрически, как ему и положено.
        """
        acceptance = self.cfg.train.get("acceptance_profile", None)
        if acceptance:
            a = list(acceptance)
            if len(a) < drafted:
                a = a + [a[-1]] * (drafted - len(a))
            a = a[:drafted]
            tail = [0.0] * (drafted + 2)
            for j in range(drafted - 1, 0, -1):
                tail[j] = a[j] * (1.0 + tail[j + 1])
            survival, prefix = [], 1.0
            for j in range(drafted):
                survival.append(prefix * (1.0 + tail[j + 1]))
                prefix *= a[j]
            w = torch.as_tensor(survival, device=like.device, dtype=torch.float32)
            return (w / w.mean()).repeat(count)[None]

        survival = self.cfg.train.get("position_weights", None)
        if not survival:
            return None
        # dtype ВСЕГДА fp32, а не `like.dtype`. Маскирующая ветвь передавала
        # сюда fp32-уверенность, симплексная — bf16-маску, и один и тот же
        # профиль округлялся по-разному: 2.824245 против 2.828125 на первой
        # позиции, расхождение до 0.82%. Веса — свойство метрики, они обязаны
        # быть у ветвей одним числом, а не почти одним.
        w = torch.as_tensor(
            list(survival)[:drafted], device=like.device, dtype=torch.float32,
        )
        if w.numel() < drafted:
            # Продолжение ГЕОМЕТРИЧЕСКОЕ, а не повтор последнего значения:
            # повтор давал хвосту постоянный вес, и при block_size=64 позиции
            # 8..63 забирали 36% всей массы вместо примерно 2%.
            ratio = (min(1.0, float(w[-1]) / float(w[-2]))
                     if w.numel() >= 2 and float(w[-2]) > 0 else 0.5)
            extra = torch.arange(
                1, drafted - w.numel() + 1, device=like.device, dtype=torch.float32,
            )
            w = torch.cat([w, float(w[-1]) * ratio ** extra])
        return (w / w.mean()).repeat(count)[None]

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
            # Data-dependent and therefore NOT synchronised across ranks: some
            # packed batches contain no document long enough for one complete
            # block. Raising here hangs DDP -- the rank that raised leaves the
            # collective while every other rank waits inside it forever, and the
            # job dies on a NCCL timeout minutes later with no useful message.
            # Return None and let the caller emit a graph-connected zero, which
            # is what the masked baseline already does for the same case.
            return None
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
            ids[:, anchors],
            df_kwargs,
        )

    def _shared_step(self, batch):
        count = int(self.cfg.train.get("anchors_per_sequence", 1))
        if count > 1:
            prepared = self._prepare_blocks(batch)
            if prepared is None:
                return None
        else:
            prepared = (*self._prepare_block(batch), {})
        (
            teacher_logits,
            block_ids,
            ctx_mask,
            block_mask,
            cache,
            anchor,
            anchor_ids,
            df_kwargs,
        ) = prepared
        x1 = self.df_processor.to_simplex(block_ids, attention_mask=block_mask)
        x_s, x_t, s, t = self.sample_trajectory(x1, block_mask)
        # Форвард на общей паре (s, t) нужен ТОЛЬКО членам CFM: log_draft входит
        # в EC, а pi -- в точку приземления x_jump. При endpoint_weight = 0 и
        # lambda = 0 его выход в лосс не входит вовсе, и градиент по нему ровно
        # нулевой (проверено retain_grad). Это был мёртвый форвард: 1 из 3 у
        # руки только с verify_kl и 1 из 7 у рук с самокоррекцией, то есть
        # 33% и 14% всей работы шага. Заодно он делал число форвардов у ветвей
        # разным (7 против 6) без всякой причины.
        draft_logits = None
        if self._needs_trajectory_forward():
            draft_logits = self._df_forward(
                x_s, anchor, ctx_mask, cache, s, t, df_kwargs
            )
        count = int(self.cfg.train.get("anchors_per_sequence", 1))
        # The decode loop enters every cycle at the pure prior, s = 0, so that
        # is the only state this term should be evaluated at.
        verify_input = self.sample_prior(x1, block_mask)
        verify_s = torch.zeros_like(s)
        ones = torch.ones_like(t)
        verify_logits = self._df_forward(
            verify_input, anchor, ctx_mask, cache, verify_s, ones, df_kwargs
        )

        shared = dict(
            teacher_logits=teacher_logits,
            draft_logits=draft_logits,
            verify_logits=verify_logits,
            x_s=x_s,
            x_t=x_t,
            x1=x1,
            s=s,
            t=t,
            ctx_mask=ctx_mask,
            block_mask=block_mask,
            cache=cache,
            anchor=anchor,
            anchor_ids=anchor_ids,
            df_kwargs=df_kwargs,
            onpolicy_logits=None,
            expected=None,
            accepted=None,
            known=None,
        )

        # ONE frozen AR forward over the drafter's own proposal, shared by every
        # term that needs either an on-policy target or the verdict. It is the
        # forward the decode loop spends on verification, and it is what the
        # objective has never contained: a target the drafter's state enters,
        # and the exact accept/reject boundary rather than a corpus-conditioned
        # guess at it.
        if self._needs_verifier_response():
            draft_ids = verify_logits.detach().argmax(-1)
            onpolicy_logits = self._verifier_response(
                draft_ids, anchor_ids, ctx_mask, cache, df_kwargs
            )
            expected = onpolicy_logits.argmax(-1)
            accepted, known = self._greedy_verdict(
                draft_ids, expected, block_mask.bool(), anchor.size(1)
            )
            shared.update(
                onpolicy_logits=onpolicy_logits,
                expected=expected,
                accepted=accepted,
                known=known,
            )
        return shared

    def _guard_compiled_ar(self):
        """Компилированный AR несовместим с он-полиси таргетом.

        `Orthrus._masked_step` зовёт замороженный ствол с
        `use_compiled_ar=True`, а `_verifier_response` — без него. Под bf16
        компилированный и eager пути расходятся на близких значениях, и тогда
        цель KL и «точный жадный вердикт» приходят из разной арифметики: член
        учил бы согласию с одной моделью, а приёмку считал бы по другой.
        Молча это не проходит.
        """
        if not self.cfg.model.backbone.get("compile_ar", False):
            return
        if self._needs_verifier_response():
            raise ValueError(
                "model.backbone.compile_ar=true is incompatible with the "
                "on-policy target: the teacher would run compiled and the "
                "verifier eager, and under bf16 the two disagree on near-ties. "
                "Set compile_ar=false or turn off train.selfcorrect_kl_weight "
                "and train.log_train_acceptance"
            )

    def _needs_trajectory_forward(self):
        """Нужен ли форвард на общей паре (s, t).

        Его выход читают только члены CFM: `log_draft` -> EC, `pi` -> точка
        приземления `x_jump` -> EC и TD, и `x_t` -> диагональ endpoint. При
        нулевых `endpoint_weight` и `lambda` в лосс не входит ничего из этого.
        Условие конфиговое, а не батчевое, поэтому одинаково на всех рангах.
        """
        if float(self.cfg.train.get("endpoint_weight",
                                    self.cfg.train.get("anchor_weight", 1.0))):
            return True
        return bool(float(self.cfg.train.get("lambda", 1.0)))

    def _needs_verifier_response(self):
        """Is any enabled term paying for the on-policy AR forward?"""
        return (
            float(self.cfg.train.get("selfcorrect_kl_weight", 0.0)) > 0.0
            or str(self.cfg.train.get("teacher_source", "corpus")) == "onpolicy"
            or bool(self.cfg.train.get("log_train_acceptance", False))
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
        """The parent's three terms in block geometry.

        Differences from the full-sequence variant: no one-position shift (teacher
        is pre-aligned in ``_shared_step``), the live mask is the block's
        own, and every DF forward carries the clean-prefix cache AND the
        clean in-block anchor (via :meth:`_df_forward`).
        """
        eps = float(self.cfg.train.get("gamma_clamp", 1e-4))
        live = block_mask.bool()
        if not live.any():
            # the whole block landed in padding: a zero step wired into the
            # graph instead of NaN from a mean over an empty tensor. Логи всё
            # равно выпускаются -- см. _log_zero_terms.
            self._log_zero_terms(metric_prefix, on_step=log_on_step,
                                 on_epoch=log_on_epoch)
            return verify_logits.sum() * 0.0
        # `verify_logits` считается всегда, поэтому графово связанный ноль
        # берётся из него: раньше он брался из `draft_logits`, которого при
        # выключенных членах CFM больше нет.
        log_draft = (F.log_softmax(draft_logits.float(), -1)
                     if draft_logits is not None else None)
        # Verifier alignment on the jump the decode loop finishes with.
        #   KL input:  π^θ_{0,1}(x_0) — the pure prior at s = 0, i.e.
        #              the state the decode loop enters every cycle at.
        #   KL target: sg(p_AR) — the frozen AR path's distribution for the
        #              same block positions, already aligned in _shared_step.
        pos_w = self._position_weights(teacher_logits, x1, live)
        verify_kl = self._teacher_loss(
            self._exact_target(teacher_logits, onpolicy_logits, known),
            verify_logits, live, position_weight=pos_w,
        )

        # Landing point of the jump — the EC-target input. Detached: the
        # jump's single teacher is ECLD.
        pi = gamma = x_jump = None
        if log_draft is not None:
            pi = log_draft.exp()
            gamma = ((t - s) / (1.0 - s).clamp(min=eps))[:, None, None]
            x_jump = x_s + gamma * (pi - x_s)
            x_jump = (x_jump * block_mask[..., None].to(x_jump.dtype)).detach()

        # --- categorical VFM endpoint likelihood on the diagonal, plus the two
        # consistency terms. Each costs a DF forward, and each was being paid
        # for unconditionally and then multiplied by a weight that is zero in
        # the measured working point -- three forwards out of eight spent on
        # terms that contribute nothing. Skip them when their weight is zero;
        # the graph-connected zero keeps every parameter in DDP's reduction.
        lam = self._lambda()
        endpoint_weight = self.cfg.train.get(
            "endpoint_weight", self.cfg.train.get("anchor_weight", 1.0)
        )
        zero = verify_logits.sum() * 0.0
        endpoint, ec, td = zero, zero, zero
        ec_kl = None
        anchor_point = self.cfg.train.get("anchor_point", "trajectory")
        diag_logits = None
        if endpoint_weight or lam:
            anchor_input = {"trajectory": x_t, "landing": x_jump}.get(anchor_point)
            if anchor_input is None:
                raise ValueError(
                    f"unknown anchor_point='{anchor_point}' (trajectory | landing)"
                )
            diag_logits = self._df_forward(
                anchor_input, anchor, ctx_mask, cache, s=t, t=t, df_kwargs=df_kwargs
            )
        if endpoint_weight:
            endpoint_nll = F.cross_entropy(
                diag_logits.float().transpose(1, 2),
                x1.argmax(-1),
                reduction="none",
            )
            endpoint = endpoint_nll[live].mean()
        if lam:
            # --- L_CE-EC — eq. (18) in "Categorical Flow Maps" (Roos et al.):
            # the jump must agree with the stop-grad expert at its landing point.
            if anchor_point == "landing":
                tgt = diag_logits.detach().float().softmax(-1)
            else:
                with torch.no_grad():
                    tgt = self._df_forward(
                        x_jump, anchor, ctx_mask, cache, s=t, t=t, df_kwargs=df_kwargs
                    ).float().softmax(-1)
            # Логируется KL, а не CE: CE = KL + H(tgt) никогда не достигает
            # нуля, поэтому записанное значение доминируется энтропией таргета
            # и с verify_kl несравнимо. Градиент тот же -- tgt отцеплен.
            #
            # Гейта по гамме здесь НЕТ, и это проверено, а не принято на веру.
            # При t -> s таргет и предсказание приходят из одного вызова сети,
            # то есть tgt = sg(p), и градиент CE по логитам равен p - sg(p) =
            # РОВНО НОЛЬ (замер: |dCE/dlogits|_inf = 1.9e-9). Вырожденные
            # розыгрыши не портят обучение, они в него просто не входят.
            # Отбрасывать их означало бы лишь домножить член на 1/P(gamma>eps)
            # = 1.19 без всякой причины.
            ce = -(tgt * log_draft).sum(-1)
            ec = ce[live].mean()
            # KL = CE - H(tgt). Считается ЗДЕСЬ и подключается к метрикам ниже:
            # значение раньше присваивалось в атрибут, который никто не читал,
            # и логировалась всё та же CE. Тензор, не float, — приведение к
            # питоновскому числу синхронизировало бы хост на каждом шаге.
            ec_kl = (ce + (tgt * tgt.clamp_min(1e-12).log()).sum(-1))[live].mean().detach()

            td = self._td_term(
                pi, gamma, s, t, live,
                forward_dt=lambda dt: self._df_forward(
                    x_s, anchor, ctx_mask, cache, s=s, t=t + dt, df_kwargs=df_kwargs
                ),
            )

        selfcorrect_kl = None
        if onpolicy_logits is not None:
            selfcorrect_kl = self._selfcorrect_kl(
                verify_logits, onpolicy_logits, accepted, known, anchor_ids,
                block_mask, ctx_mask, cache, anchor, df_kwargs,
            )
        selfcorrect_kl_weight = float(self.cfg.train.get("selfcorrect_kl_weight", 0.0))
        if selfcorrect_kl is None:
            selfcorrect_kl = zero

        verify_kl_weight = self.cfg.train.get("verify_kl_weight", 0.0)
        loss = (
            verify_kl_weight * verify_kl
            + selfcorrect_kl_weight * selfcorrect_kl
            + endpoint_weight * endpoint
            + lam * (4.0 * ec + 2.0 * td)
        )
        metrics = {
            f"{metric_prefix}/endpoint": endpoint,
            f"{metric_prefix}/verify_kl": verify_kl,
            f"{metric_prefix}/selfcorrect_kl": selfcorrect_kl,
            f"{metric_prefix}/ec": ec_kl if ec_kl is not None else ec,
            f"{metric_prefix}/td": td,
            f"{metric_prefix}/lambda": lam,
        }
        if accepted is not None:
            # The on-policy forward makes the training-time acceptance curve
            # free, and it is the metric the project actually optimises. Until
            # now it existed only as a validation decode over a couple of
            # prompts; here it is measured on every block of every step, under
            # the exact greedy rule, and it is what the objective should be
            # judged against long before a decode run happens.
            per_block = accepted.view(accepted.size(0), anchor.size(1), -1)
            metrics[f"{metric_prefix}/accepted"] = (
                per_block.sum(-1).to(torch.float32).mean()
            )
        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=log_on_epoch,
            sync_dist=True,
        )
        return loss

    def _log_zero_terms(self, metric_prefix, *, on_step, on_epoch):
        """Тот же набор ключей, что и у полного шага, но нулями.

        `self.log(..., sync_dist=True)` при `on_step=True` выполняет НАСТОЯЩИЙ
        all_reduce внутри вызова, а NCCL сопоставляет коллективы по порядку их
        выпуска. Ранний возврат на одном ранге означает, что он выпустил на
        восемь коллективов меньше, и обучение виснет либо, что хуже, ранги
        начинают спаривать чужие редукции. Набор ключей обязан совпадать на всех
        рангах, поэтому наличие `accepted` берётся из конфига
        (`_needs_verifier_response`), а не из того, что нашлось в этом батче.
        """
        zero = torch.zeros((), device=self.device)
        names = ["endpoint", "verify_kl", "selfcorrect_kl", "ec", "td", "lambda"]
        if self._needs_verifier_response():
            names.append("accepted")
        self.log_dict(
            {f"{metric_prefix}/{name}": zero for name in names},
            on_step=on_step, on_epoch=on_epoch, sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self._guard_compiled_ar()
        shared = self._shared_step(batch)
        if shared is None:
            # No usable block in this batch. Every trainable parameter must stay
            # connected or DDP's reduction hangs on the ranks that did find one
            # -- и ровно по той же причине надо выпустить те же логи.
            self._log_zero_terms("loss", on_step=True, on_epoch=False)
            self.log("train/loss", torch.zeros((), device=self.device),
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            return sum(p.sum() for p in self.orthrus.df_parameters()) * 0.0
        loss = self.compute_loss(**shared)
        self._assert_finite(loss, batch_idx)
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
            shared = self._shared_step(batch)
            if shared is None:
                # Тот же набор ключей, что и у полного прохода: `val/loss`,
                # разложение по членам, попозиционная приёмка и согласие с
                # учителем. Пропуск любого из них на одном ранге рассинхронит
                # эпохальную сводку, а `val/loss` вдобавок монитор.
                zero = torch.zeros((), device=self.device)
                self._log_zero_terms("val/loss", on_step=False, on_epoch=True)
                drafted = int(self.cfg.train.get("block_size", 8)) - 1
                self.log("val/loss", zero, prog_bar=True, on_step=False,
                         on_epoch=True, sync_dist=True)
                self.log_dict(
                    {f"val/acceptance_pos_{i + 1:02d}": zero
                     for i in range(drafted)},
                    prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
                )
                self.log("val/teacher_agreement", zero, prog_bar=True,
                         on_step=False, on_epoch=True, sync_dist=True)
                self._maybe_decode_val(batch, batch_idx)
                return None
            teacher_logits = shared["teacher_logits"]
            verify_logits = shared["verify_logits"]
            block_mask = shared["block_mask"]
            loss = self.compute_loss(
                **shared,
                metric_prefix="val/loss",
                log_on_step=False,
                log_on_epoch=True,
            )
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
