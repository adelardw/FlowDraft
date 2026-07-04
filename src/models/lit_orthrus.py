import time

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import DynamicCache
from src.models.model import build_model
from src.preprocessor import DiffusionProcessor
from transformers import AutoTokenizer

class FlowMapOrthrus(L.LightningModule):
    """Training policy around :class:`OrthrusAttentionAdapter` (the mechanism).

    This module owns everything the adapter deliberately does not:

    * the frozen AR-teacher forward (stop-gradient by ``torch.no_grad``),
    * sampling the CFM trajectory point ``(x_s, s, t)`` for the drafter,
    * the optimizer over ``df_parameters()`` (DF twins + time embedding),
    * checkpoints that store ONLY the trainable DF head — the 3B frozen
      backbone is restored from HF by ``build_model``, never written to disk.

    The dual-distillation objective lives in :meth:`compute_loss`:
    ``anchor + lambda * (4*EC + 2*TD)`` — the AR teacher anchors the diagonal
    ``π_{t,t}``, endpoint consistency propagates it to the jumps, temporal
    drift keeps the family smooth in ``t``. Knobs: ``train.lambda``,
    ``train.anchor_point``, ``train.time_sampling``.

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
        mode = self.cfg.train.get("time_sampling", "triangle")
        sampler = getattr(self, f"sample_times_{mode}", None)
        if sampler is None:
            raise ValueError(f"unknown time_sampling='{mode}' (sequential | triangle)")
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

    @staticmethod
    def _masked_kl(log_p, log_q, live):
        """``KL(p || q)`` per position, averaged over live (non-pad) tokens."""
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        return kl[live].mean()

    def compute_loss(self, batch, teacher_logits, draft_logits, x_s, x_t, s, t):
        """Dual distillation with the DIAGONAL as the AR anchor.

        Content and skill enter through different doors (this is the "dual"):

        * anchor — ``KL(sg(p_AR) || π^θ_{t,t}(·))``: the AR teacher anchors
          only the diagonal (soft target — we distill the verifier's
          distribution; the hard variant would target the actual tokens
          ``x1``, see the anchor-target design knob). Jumps get NO direct AR
          loss — such a target is constant in the noise seed at s=0,
          collapsing the transport to one point.
        * EC — ``CE(sg(π^θ_{t,t}(X_{s,t}(x_s))), π^θ_{s,t}(x_s))``: the jump
          must agree with the diagonal asked at the jump's own landing point.
          Truth flows ``p_AR -> π_{t,t} -> π_{s,t}``; at convergence landing
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

        # --- anchor: p_AR certifies the diagonal. WHERE the diagonal is
        # evaluated is an experimental knob (cfg.train.anchor_point):
        #   trajectory — KL input: π_{t,t}(x_t),          KL target: sg(p_AR(·|x1_{<i}))
        #   landing    — KL input: π_{t,t}(X_{s,t}(x_s)), KL target: sg(p_AR(·|x1_{<i}))
        # The target is identical in all modes: at position i — the frozen AR
        # path's distribution of token i given the CLEAN prefix x1_{<i}
        # (teacher_logits[:, i-1], hence the one-position shift below).
        anchor_point = self.cfg.train.get("anchor_point", "trajectory")
        anchor_input = {"trajectory": x_t, "landing": x_jump}.get(anchor_point)
        if anchor_input is None:
            raise ValueError(f"unknown anchor_point='{anchor_point}' (trajectory | landing)")
        diag_logits = self.orthrus(anchor_input, mask, use_df=True, s=t, t=t).logits
        anchor = self._masked_kl(
            F.log_softmax(teacher_logits[:, :-1].float(), -1),
            F.log_softmax(diag_logits[:, 1:].float(), -1),
            live[:, 1:],
        )

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
        dt = torch.where(t + 1e-3 <= 1.0, 1e-3, -1e-3)
        pi_dt = self.orthrus(x_s, mask, use_df=True, s=s, t=t + dt).logits.float().softmax(-1)
        drift = ((pi_dt - pi) / dt[:, None, None]).pow(2).sum(-1)  # [B, T]
        td = (gamma.squeeze(-1) ** 2 * drift)[live].mean()

        loss = anchor + self.cfg.train.get("lambda", 1.0) * (4.0 * ec + 2.0 * td)
        self.log_dict({"loss/anchor": anchor, "loss/ec": ec, "loss/td": td})
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

    def _draft_block(self, cache, block_size, times):
        """Dirichlet noise -> jump schedule via :meth:`predict` -> draft ids.

        The shared cache stays AR-only: the adapter crops the draft's K/V
        right after each forward.
        """
        x = torch.distributions.Dirichlet(
            torch.ones(self.df_processor.vocab_size, device=self.device)
        ).sample((1, block_size))
        mask = torch.ones(1, cache.get_seq_length() + block_size, dtype=torch.long, device=self.device)
        for s_i, t_i in zip(times[:-1], times[1:]):
            x = self.predict(x, mask, s_i, t_i, past_key_values=cache)
        return x.argmax(-1)

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
        **tokenizer_kwargs,
    ):
        """FULL lossless generation: draft -> verify -> commit, until done.

        Every cycle the flow map drafts ``block_size`` tokens in
        ``len(jumps)`` forwards, one AR forward verifies the block
        (longest accepted prefix + AR's own correction/bonus token), the
        cache is cropped to the accepted text and one 1-token AR forward
        commits the extra token. The output ids are bit-identical to
        :meth:`ar_generate` — the drafter affects speed, never content.

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

        while len(emitted) < max_new_tokens:
            draft_ids = self._draft_block(cache, block_size, times)
            n_forwards += len(times) - 1

            committed = cache.get_seq_length()
            mask = torch.ones(1, committed + block_size, dtype=torch.long, device=self.device)
            verify_logits = self.orthrus(draft_ids, mask, past_key_values=cache).logits
            n_forwards += 1

            n_accepted, next_token = self.verify_greedy(draft_ids, last_logits, verify_logits)
            cache.crop(committed + n_accepted)  # rejected draft K/V never pollute the cache
            acceptance.append(n_accepted)
            new = draft_ids[0, :n_accepted].tolist() + [int(next_token)]
            emitted.extend(new)
            if eos_token_id is not None and eos_token_id in new:
                break

            # commit next_token's K/V; its logits are the next cycle's last_logits
            mask = torch.ones(1, cache.get_seq_length() + 1, dtype=torch.long, device=self.device)
            out = self.orthrus(next_token.view(1, 1), mask, past_key_values=cache)
            last_logits = out.logits[:, -1]
            n_forwards += 1

        return self._finalize(input_ids, emitted, max_new_tokens, eos_token_id, start, n_forwards,
                              acceptance=acceptance)

    @torch.no_grad()
    def ar_generate(self, text=None, *, input_ids=None, max_new_tokens: int = 128,
                    eos_token_id=None, **tokenizer_kwargs):
        """Plain greedy AR generation through the frozen path — the
        correctness reference (generate() must match it token-for-token)
        and the throughput baseline (1 token per forward)."""
        input_ids = self._encode(text, input_ids, **tokenizer_kwargs)
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id

        start = time.perf_counter()
        cache = DynamicCache(config=self.orthrus.model.config)
        out = self.orthrus(input_ids, torch.ones_like(input_ids), past_key_values=cache)
        n_forwards = 1
        emitted = []

        while len(emitted) < max_new_tokens:
            token = out.logits[:, -1].argmax(-1)
            emitted.append(int(token))
            if eos_token_id is not None and int(token) == eos_token_id:
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
        with torch.no_grad():
            teacher_logits = self.orthrus(ids, mask).logits
        x_s, x_t, s, t = self.sample_trajectory(x1, mask)
        draft_logits = self.orthrus(x_s, mask, use_df=True, s=s, t=t).logits
        return teacher_logits, draft_logits, x_s, x_t, s, t

    def training_step(self, batch, batch_idx):
        teacher_logits, draft_logits, x_s, x_t, s, t = self._shared_step(batch)
        loss = self.compute_loss(batch, teacher_logits, draft_logits, x_s, x_t, s, t)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        teacher_logits, draft_logits, x_s, x_t, s, t = self._shared_step(batch)
        loss = self.compute_loss(batch, teacher_logits, draft_logits, x_s, x_t, s, t)
        # Cheap acceptance proxy: how often the drafter's argmax already
        # matches the verifier's argmax (shifted: teacher@i predicts i+1).
        mask = batch["attention_mask"][:, 1:].bool()
        agree = (draft_logits[:, 1:].argmax(-1) == teacher_logits[:, :-1].argmax(-1))[mask]
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/teacher_agreement", agree.float().mean())
        return loss

    def configure_optimizers(self):
        opt = self.cfg.train
        return torch.optim.AdamW(
            self.orthrus.df_parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            betas=tuple(opt.betas),
        )

    def on_save_checkpoint(self, checkpoint):
        # Keep only the DF head (~1.7 GB for 3B instead of ~14 GB with the
        # frozen backbone); build_model restores the backbone from HF on load.
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            key: value for key, value in checkpoint["state_dict"].items() if key in trainable
        }