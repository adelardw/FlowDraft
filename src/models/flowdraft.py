import math
import time

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import DynamicCache
from src.models.model import build_model
from src.preprocessor import DiffusionProcessor
from transformers import AutoTokenizer

class FlowDraft(L.LightningModule):
    """Training policy around :class:`FlowDraftAttentionAdapter` (the mechanism).

    This module owns everything the adapter deliberately does not:

    * the optional frozen AR-teacher forward (stop-gradient by ``torch.no_grad``),
    * sampling the CFM trajectory point ``(x_s, s, t)`` for the drafter,
    * the optimizer over ``df_parameters()`` (DF twins + time embedding),
    * checkpoints that store ONLY the trainable DF head — the 3B frozen
      backbone is restored from HF by ``build_model``, never written to disk.

    The categorical flow-map objective lives in :meth:`compute_loss`:
    ``endpoint + lambda * (4*EC + 2*TD)`` — categorical VFM anchors the
    diagonal ``π_{t,t}``, endpoint consistency propagates it to the jumps,
    and temporal drift keeps the family smooth in ``t``. An optional,
    separately weighted AR KL auxiliary can align the diagonal with the
    verifier for experimental runs.

    Expected batch — a dict with:
        ``input_ids [B, T]`` (long) · ``attention_mask [B, T]`` (long, 1=live)
        · optionally ``simplex [B, T, V]`` (built on-device from input_ids
        when absent — the recommended mode, [B, T, 128k] must not ride the
        DataLoader).
    """

    def __init__(self, cfg, orthrus=None, tokenizer: AutoTokenizer =None, df_processor : DiffusionProcessor=None):
        super().__init__()
        self.cfg = cfg
        if orthrus is None:
            orthrus, tokenizer, df_processor = build_model(cfg.model)
        self.orthrus = orthrus
        self.tokenizer = tokenizer
        self.df_processor = df_processor
        # Checkpoints hold only the DF head (see on_save_checkpoint), so the
        # frozen backbone keys are legitimately absent on load.
        self.strict_loading = False
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    # --- mechanism passthrough ------------------------------------------------

    def forward(self, *args, **kwargs):
        """Delegate to the adapter: ``forward(ids_or_simplex, mask, use_df=..., s=..., t=...)``."""
        return self.orthrus(*args, **kwargs)

    # --- CFM trajectory (design knob: override to try other schedules) --------

    def sample_times_sequential(self, batch, device):
        """``s ~ U[0, 1)``, then ``t ~ U[s, 1]``.

        Pair density on the triangle is ∝ 1/(1 - s): late, short jumps are
        overweighted relative to early ones.
        """
        s = torch.rand(batch, device=device)
        t = s + (1.0 - s) * torch.rand(batch, device=device)
        return s, t

    def sample_times_triangle(self, batch, device):
        """Uniform on the triangle ``{0 <= s <= t <= 1}``.

        Two ``U[0, 1]`` draws, sorted: every (s, t) pair is equally likely —
        no bias toward any jump length or start time.
        """
        u = torch.rand(batch, 2, device=device)
        return u.min(dim=-1).values, u.max(dim=-1).values

    def sample_times_paper(self, batch, device):
        """``t ~ U[0, 1]``, then ``s ~ U[0, t]``.

        Pair density on the triangle is ∝ 1/t: short EARLY jumps are
        overweighted, and the anchor's diagonal times are uniform
        (E[t] = 1/2 vs 2/3 for triangle, 3/4 for sequential).
        """
        t = torch.rand(batch, device=device)
        s = t * torch.rand(batch, device=device)
        return s, t

    def sample_trajectory(self, simplex, attention_mask=None):
        """Draw one training point of the linear simplex path per sample.

        ``x0 ~ Dirichlet(1)`` (uniform on the simplex), ``x1`` = the clean
        one-hot endpoints from the batch, ``x_s = (1 - s) x0 + s x1``. The
        pair ``(s, t)`` comes from ``sample_times_<cfg.train.time_sampling>``
        so the whole two-time family gets signal: ``t = s`` is the diagonal
        (the local VFM expert ``π_{t,t}`` that ECLD distills from), ``t = 1``
        is the full drafting jump, everything in between feeds consistency.
        Pad rows are zeroed — the same off-simplex sentinel the processor
        uses; they are masked out of attention and must not enter the loss.
        """
        batch = simplex.size(0)
        mode = self.cfg.train.get("time_sampling", "paper")
        sampler = getattr(self, f"sample_times_{mode}", None)
        if sampler is None:
            raise ValueError(f"unknown time_sampling='{mode}' (sequential | triangle | paper)")
        s, t = sampler(batch, simplex.device)
        x0 = torch.distributions.Dirichlet(
            torch.ones(simplex.size(-1), device=simplex.device)
        ).sample(simplex.shape[:2])
        x_s = (1.0 - s[:, None, None]) * x0 + s[:, None, None] * simplex
        # Same trajectory at time t — the anchor input. Built from DATA, so it
        # is correct from step one (a landing-point anchor would bootstrap on
        # the untrained jump's garbage and chase a θ-dependent distribution).
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * simplex
        if attention_mask is not None:
            pad = attention_mask[..., None].to(x_s.dtype)
            x_s = x_s * pad
            x_t = x_t * pad
        return x_s, x_t, s, t

    # --- the loss is yours ----------------------------------------------------

    def _lambda(self):
        """ECLD weight with optional staging: endpoint inference FIRST, then
        consistency. With ``train.lambda_ramp_steps = N > 0`` the weight ramps
        linearly ``0 -> train.lambda`` over the first N optimizer steps, so
        early training is anchor-only (the diagonal must be trustworthy
        before it can teach the jumps); ``0`` = static from step one.
        """
        lam = self.cfg.train.get("lambda", 1.0)
        ramp = int(self.cfg.train.get("lambda_ramp_steps", 0))
        if ramp > 0:
            lam = lam * min(1.0, self.global_step / ramp)
        return lam

    @staticmethod
    def _masked_kl(log_p, log_q, live):
        """``KL(p || q)`` per position, averaged over live (non-pad) tokens.

        An empty ``live`` mask yields a graph-connected zero, not NaN
        (``mean()`` over an empty tensor silently poisons the weights).
        """
        if not live.any():
            return log_q.sum() * 0.0
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        return kl[live].mean()

    def compute_loss(self, batch, teacher_logits, draft_logits, x_s, x_t, x1, s, t):
        """Categorical flow-map training plus optional AR distillation.

        The paper's diagonal VFM objective and off-diagonal ECLD objective are
        kept distinct:

        * endpoint — ``CE(x1, π^θ_{t,t}(x_t))``: the categorical VFM endpoint
          likelihood that makes the diagonal the denoiser associated with the
          sampled interpolation. This is the anchor required by CFM theory.
        * AR KL (optional) — ``KL(sg(p_AR) || π^θ_{t,t})``: a separately
          weighted verifier-alignment auxiliary. It defaults off because it
          is not part of the CFM objective and can conflict with the sampled
          endpoint at a given training trajectory.
        * EC — ``CE(sg(π^θ_{t,t}(X_{s,t}(x_s))), π^θ_{s,t}(x_s))``: the jump
          must agree with the diagonal asked at the jump's own landing point.
          Truth flows ``x1 -> π_{t,t} -> π_{s,t}``; at convergence landing
          points are distributed like ``x_t``, so the two diagonals coincide.
        * TD — temporal drift ``||∂_t π^θ_{s,t}||²``, finite difference in
          the time INPUT.

        ``teacher_logits[:, i]`` is the AR distribution of token ``i + 1``,
        hence the shift in the anchor. Masking happens only here, at the
        reductions — padding must not contribute to any mean.
        """
        eps = 1e-4
        mask = batch["attention_mask"]
        live = mask.bool()
        log_draft = F.log_softmax(draft_logits.float(), -1)

        # Landing point of the jump — always needed as the EC-target input.
        # Detached: the jump's single teacher is ECLD, and the expert is not
        # trained THROUGH jump-generated inputs.
        pi = log_draft.exp()
        gamma = ((t - s) / (1.0 - s).clamp(min=eps))[:, None, None]
        x_jump = x_s + gamma * (pi - x_s)
        x_jump = (x_jump * mask[..., None].to(x_jump.dtype)).detach()  # pads stay zero

        # --- diagonal endpoint inference. The paper-faithful setting is
        # ``trajectory``: π_{t,t}(x_t) is supervised with the categorical
        # endpoint x1 used to construct that same interpolant. ``landing`` is
        # retained only as an explicitly experimental off-trajectory option.
        anchor_point = self.cfg.train.get("anchor_point", "trajectory")
        anchor_input = {"trajectory": x_t, "landing": x_jump}.get(anchor_point)
        if anchor_input is None:
            raise ValueError(f"unknown anchor_point='{anchor_point}' (trajectory | landing)")
        diag_logits = self.orthrus(anchor_input, mask, use_df=True, s=t, t=t).logits
        endpoint_nll = F.cross_entropy(
            diag_logits[:, 1:].float().transpose(1, 2),
            x1[:, 1:].argmax(-1),
            reduction="none",
        )
        endpoint_live = live[:, 1:]
        endpoint = (
            endpoint_nll[endpoint_live].mean()
            if endpoint_live.any()
            else diag_logits.sum() * 0.0
        )
        ar_kl_weight = self.cfg.train.get("ar_kl_weight", 0.0)
        if ar_kl_weight:
            if teacher_logits is None:
                raise ValueError("teacher logits are required when train.ar_kl_weight > 0")
            ar_kl = self._masked_kl(
                F.log_softmax(teacher_logits[:, :-1].float(), -1),
                F.log_softmax(diag_logits[:, 1:].float(), -1),
                live[:, 1:],
            )
        else:
            ar_kl = diag_logits.sum() * 0.0

        # --- L_CE-EC — eq. (18) in "Categorical Flow Maps" (Roos et al.):
        # the jump must agree with the (stop-grad) expert
        # asked at its own landing point X_{s,t}(x_s) — a level-t input.
        # In landing mode the anchor forward already evaluated the expert
        # there, so its detached logits are reused (one forward saved).
        # w_t ≡ 1: the bound is stated with (1-t)^{-2}, but it blows up at
        # t→1; the authors report w ≡ 1 as the most stable choice.
        if anchor_point == "landing":
            tgt = diag_logits.detach().float().softmax(-1)
        else:
            with torch.no_grad():
                tgt = self.orthrus(x_jump, mask, use_df=True, s=t, t=t).logits.float().softmax(-1)
        ec = -(tgt * log_draft).sum(-1)[live].mean()

        # --- L_TD — eq. (16) in "Categorical Flow Maps" (Roos et al.):
        # ||∂_t π^θ_{s,t}(x_s)||², finite difference in t
        # (∂_t is a derivative w.r.t. the time INPUT — autograd .grad is not it).
        dt_val = 0.05
        dt = torch.where(t + dt_val <= 1.0, dt_val, -dt_val)
        pi_dt = self.orthrus(x_s, mask, use_df=True, s=s, t=t + dt).logits.float().softmax(-1)
        drift = ((pi_dt - pi) / dt[:, None, None]).pow(2).sum(-1)  # [B, T]
        td = (gamma.squeeze(-1) ** 2 * drift)[live].mean()

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

    # --- generation: the model generates, start to finish ----------------------
    # src/eval.py drives these: it compares generate() vs ar_generate() and
    # asserts losslessness.

    @staticmethod
    def _jump_schedule(jumps):
        """int n -> n equal jumps over linspace(0, 1); list -> validated as-is."""
        times = torch.linspace(0, 1, jumps + 1).tolist() if isinstance(jumps, int) else list(jumps)
        if times[0] != 0 or times[-1] != 1 or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError(f"jump schedule must increase from 0 to 1, got {times}")
        return times

    @staticmethod
    def verify_greedy(draft_ids, last_logits, verify_logits):
        """Greedy lossless verification of one drafted block (batch size 1).

        The drafted token at position j is accepted iff it equals the token
        greedy AR would have produced there itself; the AR expectation for j
        is conditioned on the drafted tokens before j, so one mismatch
        invalidates everything after it — hence the longest-prefix rule.

        Args:
            draft_ids:     ``[1, K]`` — drafter proposal.
            last_logits:   ``[1, V]`` — AR distribution of the FIRST drafted
                position (from the previous cycle / prefill).
            verify_logits: ``[1, K, V]`` — AR forward over ``draft_ids``;
                position ``j`` holds the AR distribution of position ``j+1``.

        Returns ``(n_accepted, next_token)``: the accepted-prefix length and
        the token AR emits after it — its own correction at the first
        mismatch, or the bonus continuation when the whole block matched.
        Emitting it makes every cycle produce >= 1 token and keeps the
        output bit-identical to greedy AR.
        """
        expected = torch.cat(
            [last_logits.argmax(-1, keepdim=True), verify_logits[:, :-1].argmax(-1)],
            dim=1,
        )
        n_accepted = int((draft_ids == expected).cumprod(dim=1).sum())
        if n_accepted == draft_ids.size(1):
            next_token = verify_logits[:, -1].argmax(-1)  # bonus: AR's continuation
        else:
            next_token = expected[:, n_accepted]  # correction: what AR wanted instead
        return n_accepted, next_token

    def _encode(self, text, input_ids, **tokenizer_kwargs):
        if (text is None) == (input_ids is None):
            raise ValueError("pass exactly one of text / input_ids")
        if text is not None:
            enc = self.df_processor(text, return_simplex=False, **tokenizer_kwargs)
            input_ids = enc["input_ids"]
        input_ids = input_ids.to(self.device)
        assert input_ids.dim() == 2 and input_ids.size(0) == 1, "generation is batch-size-1"
        return input_ids

    @staticmethod
    def _target_probs(logits, temperature: float, top_k=None, top_p=None):
        """The AR target distribution ``p`` under the sampling params.

        ``temperature=0`` -> a delta at the argmax (greedy). ``top_k``/
        ``top_p`` filter BEFORE the softmax; the same ``p`` is used both to
        sample in :meth:`ar_generate` and to accept/reject drafted tokens,
        which is exactly what makes speculative sampling lossless in
        distribution for ANY proposal q.
        """
        logits = logits.float()
        if temperature <= 0:
            return F.one_hot(logits.argmax(-1), logits.size(-1)).float()
        logits = logits / temperature
        if top_k:
            kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p:
            sorted_logits, idx = logits.sort(dim=-1, descending=True)
            cum = sorted_logits.softmax(-1).cumsum(-1)
            drop_sorted = cum - sorted_logits.softmax(-1) > top_p  # keep first token past p
            drop = torch.zeros_like(drop_sorted).scatter(-1, idx, drop_sorted)
            logits = logits.masked_fill(drop, float("-inf"))
        return logits.softmax(-1)

    @staticmethod
    def _gumbel(seed: int, position: int, vocab: int, device):
        """Position-keyed Gumbel noise — the coupling that makes T>0 BIT-exact.

        Gumbel-max: ``argmax(log p + g)`` is an exact sample from ``p``. With
        ``g`` a deterministic function of ``(seed, generated-token index)``,
        sampling becomes a deterministic map — the AR path and the
        speculative path perturb the SAME target distributions with the SAME
        noise, so greedy-consensus verification reproduces AR sampling
        token-for-token.
        """
        gen = torch.Generator().manual_seed(seed * 1_000_003 + position)
        u = torch.rand(vocab, generator=gen).clamp(1e-9, 1 - 1e-9)
        return (-torch.log(-torch.log(u))).to(device)

    def _verify_speculative(self, draft_ids, q, last_logits, verify_logits,
                            temperature, top_k, top_p):
        """Leviathan-style accept/reject — lossless IN DISTRIBUTION.

        Token ``x_j ~ q_j`` is accepted with probability ``min(1, p_j(x_j) /
        q_j(x_j))``; at the first rejection the replacement is drawn from the
        residual ``norm(max(0, p_j − q_j))``; a fully accepted block earns a
        bonus token from ``p_K``. The drafter's quality only moves the
        acceptance rate, never the output distribution.
        """
        p = self._target_probs(
            torch.cat([last_logits[:, None], verify_logits[:, :-1]], dim=1),
            temperature, top_k, top_p,
        )  # [1, K, V]: the target distribution of every drafted position
        q = q.float()
        for j in range(draft_ids.size(1)):
            token = draft_ids[0, j]
            ratio = p[0, j, token] / q[0, j, token].clamp_min(1e-12)
            if torch.rand((), device=draft_ids.device) < ratio:
                continue
            residual = (p[0, j] - q[0, j]).clamp_min(0)
            residual = residual / residual.sum().clamp_min(1e-12)
            return j, torch.multinomial(residual, 1)
        bonus = self._target_probs(verify_logits[:, -1], temperature, top_k, top_p)
        return draft_ids.size(1), torch.multinomial(bonus[0], 1)

    def _draft_block(self, cache, block_size, times, sample: bool = False, anchor_token=None):
        """Dirichlet noise -> jump schedule via :meth:`predict`.

        The final simplex point IS the proposal distribution ``q`` (a convex
        mix of distributions stays on the simplex): greedy takes its argmax,
        sampling draws from it. The shared cache stays AR-only: the adapter
        crops the draft's K/V right after each forward.

        ``anchor_token`` — the previous cycle's correction/bonus token whose
        K/V are NOT yet in the cache. It rides as a CLEAN in-block position 0
        (the drafter sees it bidirectionally) and is re-clamped to its
        one-hot after every jump: the position is already at t=1 while the
        time labels cover the whole block (diffusion-inpainting clamp). Its
        K/V get committed by the NEXT verify forward, not by a standalone
        1-token pass — that keeps the cycle at ``jumps + 1`` forwards.

        Returns ``(draft_ids [1, K], q [1, K, V])`` for the K FRESH positions.
        """
        vocab = self.df_processor.vocab_size
        x = torch.distributions.Dirichlet(
            torch.ones(vocab, device=self.device)
        ).sample((1, block_size))
        anchor = None
        if anchor_token is not None:
            anchor = F.one_hot(anchor_token.view(1, 1), vocab).to(x.dtype)
            x = torch.cat([anchor, x], dim=1)
        mask = torch.ones(1, cache.get_seq_length() + x.size(1), dtype=torch.long, device=self.device)
        for s_i, t_i in zip(times[:-1], times[1:]):
            x = self.predict(x, mask, s_i, t_i, past_key_values=cache)
            if anchor is not None:
                x = torch.cat([anchor, x[:, 1:]], dim=1)  # keep the clean position clean
        fresh = x[:, 1:] if anchor is not None else x
        q = fresh.clamp_min(0)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)  # float-noise safety
        ids = torch.multinomial(q[0], 1).view(1, -1) if sample else q.argmax(-1)
        return ids, q

    @torch.no_grad()
    def generate(
        self,
        text=None,
        *,
        input_ids=None,
        block_size: int = 8,
        jumps=1,
        max_new_tokens: int = 128,
        eos_token_id=None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        coupled: bool = True,
        sampling_seed: int = 0,
        **tokenizer_kwargs,
    ):
        """FULL lossless generation: draft -> verify -> commit, until done.

        Every cycle: the flow map drafts ``block_size`` fresh tokens in
        ``jumps`` forwards (the previous cycle's correction/bonus token rides
        along as a clean in-block anchor), then ONE AR forward verifies the
        block — committing the anchor's K/V and scoring every draft position
        in the same pass. Cycle cost = ``jumps + 1`` forwards, nothing else.
        The drafter affects speed, never content:

        * ``temperature=0`` (default) — greedy verification; the output ids
          are BIT-identical to greedy :meth:`ar_generate`.
        * ``temperature>0, coupled=True`` (default) — Gumbel-coupled
          sampling: position-keyed Gumbel noise (``sampling_seed``) makes
          sampling a deterministic argmax, so the output is BIT-identical to
          :meth:`ar_generate` with the same temperature/seed. Different
          seeds -> different (correctly distributed) samples.
        * ``temperature>0, coupled=False`` — Leviathan speculative sampling:
          lossless IN DISTRIBUTION (equality of laws, not of tokens).

        Returns a dict: ``sequences [1, T+N]``, ``new_tokens`` (list),
        ``text`` (when a tokenizer is attached), ``acceptance`` (per cycle),
        ``n_forwards``, ``seconds``.
        """
        input_ids = self._encode(text, input_ids, **tokenizer_kwargs)
        times = self._jump_schedule(jumps)
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id

        start = time.perf_counter()
        cache = DynamicCache(config=self.orthrus.model.config)
        out = self.orthrus(input_ids, torch.ones_like(input_ids), past_key_values=cache)
        last_logits = out.logits[:, -1]
        n_forwards = 1
        emitted, acceptance = [], []
        # The correction/bonus token is NOT committed by its own 1-token pass
        # (that would make the cycle jumps+2 forwards and depress TPF).
        # Instead it stays "pending": the drafter sees it as a clean in-block
        # anchor, and the next verify forward commits its K/V and yields the
        # first fresh position's target in one go — the cycle is jumps+1.
        pending = None

        # Materialise the first token directly from the already-computed
        # prefill distribution. It becomes the clean, uncommitted anchor for
        # the first draft cycle, matching block-wise training and the Orthrus
        # inference geometry without costing another forward pass.
        if max_new_tokens > 0:
            first_probs = self._target_probs(last_logits, temperature, top_k, top_p)
            if temperature > 0 and coupled:
                first_gumbel = self._gumbel(
                    sampling_seed, 0, first_probs.size(-1), first_probs.device
                )
                pending = (
                    first_probs[0].clamp_min(1e-30).log() + first_gumbel
                ).argmax().view(1)
            elif temperature > 0:
                pending = torch.multinomial(first_probs[0], 1)
            else:
                pending = first_probs.argmax(-1)
            emitted.append(int(pending))
            if eos_token_id is not None and int(pending) == eos_token_id:
                return self._finalize(
                    input_ids, emitted, max_new_tokens, eos_token_id, start,
                    n_forwards, acceptance=acceptance,
                )

        while len(emitted) < max_new_tokens:
            draft_ids, q = self._draft_block(
                cache, block_size, times,
                sample=temperature > 0 and not coupled, anchor_token=pending,
            )
            n_forwards += len(times) - 1
            if temperature > 0 and coupled:
                # keys = indices of the tokens these positions would emit;
                # g_all[j] targets generated token (len(emitted) + j)
                base = len(emitted)
                g_all = torch.stack([
                    self._gumbel(sampling_seed, base + j, q.size(-1), q.device)
                    for j in range(draft_ids.size(1) + 1)
                ])
                # best draft = argmax of the PERTURBED proposal (same noise
                # the verifier will apply to the target distribution)
                draft_ids = (q[0].clamp_min(1e-30).log() + g_all[:-1]).argmax(-1)[None]

            committed = cache.get_seq_length()
            verify_in = draft_ids if pending is None else torch.cat(
                [pending.view(1, 1), draft_ids], dim=1
            )
            mask = torch.ones(1, committed + verify_in.size(1), dtype=torch.long, device=self.device)
            logits = self.orthrus(verify_in, mask, past_key_values=cache).logits
            n_forwards += 1
            if pending is not None:
                last_logits, verify_logits = logits[:, 0], logits[:, 1:]
            else:
                verify_logits = logits  # last_logits carried from the prefill

            if temperature > 0 and coupled:
                # Gumbel-max turns sampling into a deterministic argmax over
                # perturbed logits — the greedy-consensus machinery then
                # verifies it BIT-exactly. Perturb each target distribution
                # with the gumbel of the token index it decides.
                last_pert = (
                    self._target_probs(last_logits, temperature, top_k, top_p)
                    .clamp_min(1e-30).log() + g_all[0]
                )
                vl_pert = (
                    self._target_probs(verify_logits, temperature, top_k, top_p)
                    .clamp_min(1e-30).log() + g_all[1:][None]
                )
                n_accepted, next_token = self.verify_greedy(draft_ids, last_pert, vl_pert)
            elif temperature > 0:
                n_accepted, next_token = self._verify_speculative(
                    draft_ids, q, last_logits, verify_logits, temperature, top_k, top_p
                )
            else:
                n_accepted, next_token = self.verify_greedy(draft_ids, last_logits, verify_logits)
            # keep the (now committed) pending token + the accepted prefix;
            # rejected draft K/V never pollute the cache
            cache.crop(committed + (0 if pending is None else 1) + n_accepted)
            acceptance.append(n_accepted)
            new = draft_ids[0, :n_accepted].tolist() + [int(next_token)]
            emitted.extend(new)
            pending = next_token
            if eos_token_id is not None and eos_token_id in new:
                break

        return self._finalize(input_ids, emitted, max_new_tokens, eos_token_id, start, n_forwards,
                              acceptance=acceptance)

    @torch.no_grad()
    def ar_generate(self, text=None, *, input_ids=None, max_new_tokens: int = 128,
                    eos_token_id=None, temperature: float = 0.0,
                    top_k: int | None = None, top_p: float | None = None,
                    coupled: bool = True, sampling_seed: int = 0,
                    **tokenizer_kwargs):
        """Plain AR generation through the frozen path — the correctness
        reference and the throughput baseline (1 token per forward).
        ``temperature=0`` = greedy (bitwise reference). ``temperature>0,
        coupled=True`` = Gumbel-max sampling with position-keyed noise —
        the bitwise reference for coupled generate(); ``coupled=False`` =
        plain multinomial sampling (reference in distribution)."""
        input_ids = self._encode(text, input_ids, **tokenizer_kwargs)
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id

        start = time.perf_counter()
        cache = DynamicCache(config=self.orthrus.model.config)
        out = self.orthrus(input_ids, torch.ones_like(input_ids), past_key_values=cache)
        n_forwards = 1
        emitted = []

        while len(emitted) < max_new_tokens:
            probs = self._target_probs(out.logits[:, -1], temperature, top_k, top_p)
            if temperature > 0 and coupled:
                g = self._gumbel(sampling_seed, len(emitted), probs.size(-1), probs.device)
                token = (probs[0].clamp_min(1e-30).log() + g).argmax().view(1)
            elif temperature > 0:
                token = torch.multinomial(probs[0], 1)
            else:
                token = probs.argmax(-1)
            emitted.append(int(token))
            # no trailing forward after the LAST token: N tokens cost exactly
            # prefill + (N-1) passes, so TPF_ar == 1.0, not N/(N+1)
            if len(emitted) >= max_new_tokens or (
                eos_token_id is not None and int(token) == eos_token_id
            ):
                break
            mask = torch.ones(1, cache.get_seq_length() + 1, dtype=torch.long, device=self.device)
            out = self.orthrus(token.view(1, 1), mask, past_key_values=cache)
            n_forwards += 1

        return self._finalize(input_ids, emitted, max_new_tokens, eos_token_id, start, n_forwards)

    def _finalize(self, input_ids, emitted, max_new_tokens, eos_token_id, start, n_forwards,
                  acceptance=None):
        emitted = emitted[:max_new_tokens]
        if eos_token_id is not None and eos_token_id in emitted:
            emitted = emitted[: emitted.index(eos_token_id) + 1]
        result = {
            "sequences": torch.cat(
                [input_ids, torch.tensor([emitted], device=input_ids.device)], dim=1
            ),
            "new_tokens": emitted,
            "n_forwards": n_forwards,
            "seconds": time.perf_counter() - start,
        }
        if acceptance is not None:
            result["acceptance"] = acceptance
        if self.tokenizer is not None:
            result["text"] = self.tokenizer.decode(emitted, skip_special_tokens=True)
        return result

    def predict(self, x_s, mask, s, t, past_key_values=None):
        """One flow-map jump: the point ``X_{s,t}(x_s)`` on the simplex.

        The network's logits parametrise the endpoint distribution
        ``π^θ_{s,t}(x_s) = softmax(logits)``; the linear VFM decoder turns it
        into the jump ``X_{s,t}(x) = x + γ (π - x)`` with
        ``γ = (t - s)/(1 - s)``. At ``(s=0, t=1)`` γ = 1 and the jump is π
        itself — no special case needed.
        """
        logits = self(x_s, mask, use_df=True, s=s, t=t, past_key_values=past_key_values).logits
        pi = logits.float().softmax(-1)
        s = torch.as_tensor(s, dtype=pi.dtype, device=pi.device).reshape(-1, 1, 1)
        t = torch.as_tensor(t, dtype=pi.dtype, device=pi.device).reshape(-1, 1, 1)
        gamma = (t - s) / (1.0 - s).clamp(min=1e-4)
        return x_s + gamma * (pi - x_s)

    def _shared_step(self, batch):
        ids, mask = batch["input_ids"], batch["attention_mask"]
        x1 = batch.get("simplex")
        if x1 is None:
            # Build the one-hot endpoints HERE, on-device. A [B, T, V] tensor
            # with V≈128k is gigabytes — it must never ship through the
            # DataLoader; batches only need input_ids + attention_mask.
            x1 = self.df_processor.to_simplex(ids, attention_mask=mask)
        teacher_logits = None
        # The frozen teacher is unnecessary for paper-faithful endpoint
        # training. Keep it for validation metrics and optional AR-KL runs.
        if not self.training or self.cfg.train.get("ar_kl_weight", 0.0):
            with torch.no_grad():
                teacher_logits = self.orthrus(ids, mask).logits
        x_s, x_t, s, t = self.sample_trajectory(x1, mask)
        draft_logits = self.orthrus(x_s, mask, use_df=True, s=s, t=t).logits
        return teacher_logits, draft_logits, x_s, x_t, x1, s, t

    def training_step(self, batch, batch_idx):
        teacher_logits, draft_logits, x_s, x_t, x1, s, t = self._shared_step(batch)
        loss = self.compute_loss(batch, teacher_logits, draft_logits, x_s, x_t, x1, s, t)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {batch_idx}: {loss}")
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        teacher_logits, draft_logits, x_s, x_t, x1, s, t = self._shared_step(batch)
        loss = self.compute_loss(batch, teacher_logits, draft_logits, x_s, x_t, x1, s, t)
        # Cheap acceptance proxy: how often the drafter's argmax already
        # matches the verifier's argmax (shifted: teacher@i predicts i+1).
        mask = batch["attention_mask"][:, 1:].bool()
        agree = (draft_logits[:, 1:].argmax(-1) == teacher_logits[:, :-1].argmax(-1))[mask]
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/teacher_agreement", agree.float().mean(), sync_dist=True)
        self._maybe_decode_val(batch, batch_idx)
        return loss

    def on_validation_epoch_start(self):
        # Decode metrics are intentionally sampled from several consecutive
        # validation batches. ``data.batch_size=1`` is required by the paper
        # recipe, so restricting decoding to batch zero would otherwise turn
        # any requested sample count into a single prompt.
        self._val_decode_remaining = self.cfg.train.get("val_decode_prompts", 0)
        self._val_decode_accs = []
        self._val_decode_tpfs = []

    @torch.no_grad()
    def _maybe_decode_val(self, batch, batch_idx):
        """The REAL target metrics as validation curves: run the lossless
        decode loop (single-jump — the headline configuration) on a few val
        prompts and log ``val/acceptance_decode`` and ``val/tpf``. This is
        what training should improve to beat the baseline, and what the
        checkpoint monitor tracks; ``train.val_decode_prompts=0`` disables.
        """
        remaining = getattr(self, "_val_decode_remaining", 0)
        if remaining <= 0:
            return
        max_new = self.cfg.train.get("val_decode_max_new", 32)
        block = self.cfg.train.get("block_size", 8)
        accs, tpfs = [], []
        decoded = 0
        for i in range(min(remaining, batch["input_ids"].size(0))):
            live = int(batch["attention_mask"][i].sum())
            plen = min(max(live // 2, 2), 32)
            if live < plen + 2:
                continue
            out = self.generate(
                input_ids=batch["input_ids"][i : i + 1, :plen],
                block_size=block, jumps=1, max_new_tokens=max_new,
            )
            if out["acceptance"]:
                accs.append(sum(out["acceptance"]) / len(out["acceptance"]))
            tpfs.append(len(out["new_tokens"]) / out["n_forwards"])
            decoded += 1
        self._val_decode_remaining -= decoded
        self._val_decode_accs.extend(accs)
        self._val_decode_tpfs.extend(tpfs)
        if self._val_decode_remaining == 0 and self._val_decode_tpfs:
            if self._val_decode_accs:
                self.log(
                    "val/acceptance_decode",
                    sum(self._val_decode_accs) / len(self._val_decode_accs),
                    sync_dist=True,
                )
            self.log("val/tpf", sum(self._val_decode_tpfs) / len(self._val_decode_tpfs), sync_dist=True)

    def configure_optimizers(self):
        cfg = self.cfg.train
        optimizer = torch.optim.AdamW(
            self.orthrus.df_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=tuple(cfg.betas),
        )
        schedule = cfg.get("lr_schedule", "cosine")
        if schedule == "constant":
            return optimizer
        if schedule != "cosine":
            raise ValueError(f"unknown lr_schedule='{schedule}' (constant | cosine)")
        total = self.trainer.estimated_stepping_batches
        if not math.isfinite(total):
            # streaming dataset with no step bound: the cosine horizon is
            # undefined — the schedule must know when training ends
            raise ValueError(
                "lr_schedule=cosine needs a finite training length: set "
                "trainer.max_steps or trainer.limit_train_batches (+ max_epochs), "
                "or switch to train.lr_schedule=constant"
            )
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total * cfg.get("warmup_ratio", 0.05))),
            num_training_steps=int(total),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        # Keep only the DF head (~1.7 GB for 3B instead of ~14 GB with the
        # frozen backbone); build_model restores the backbone from HF on load.
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            key: value for key, value in checkpoint["state_dict"].items() if key in trainable
        }
