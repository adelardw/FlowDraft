import math
import time
from contextlib import contextmanager

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint
from loguru import logger
from omegaconf import OmegaConf
from transformers import DynamicCache
from src.models.model import build_model
from src.preprocessor import DiffusionProcessor
from transformers import AutoTokenizer

class FlowDraft(L.LightningModule):
    """Training policy around :class:`FlowDraftAttentionAdapter` (the mechanism).

    This module owns everything the adapter deliberately does not:

    * the optional frozen AR-teacher forward (stop-gradient by ``torch.no_grad``),
    * the prior the decode loop and the training terms start from,
    * the decode loop itself, greedy and sampled, and validation by decoding,
    * the optimizer over ``df_parameters()`` (DF twins + time embedding),
    * checkpoints that store ONLY the trainable DF head — the 3B frozen
      backbone is restored from HF by ``build_model``, never written to disk.

    It carries no loss of its own. Training lives in the block-wise subclass,
    where a verifier term aligns the one-jump map ``π_{0,1}`` — the map the
    decode loop actually executes — with the frozen AR distribution, and a
    refinement term trains the drafter on its own decode chain. The
    full-sequence objective that used to live here, ``endpoint + lambda *
    (4*EC + 2*TD)``, never carried weight in any preset and is in
    ``bucket/cfm_terms``.

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
        # FlowDraft uses Dirichlet simplex noise, not the masked baseline's
        # learned mask token. Leaving this parameter trainable makes it unused
        # in the backward graph and breaks DDP with find_unused_parameters=False.
        self.orthrus.mask_embedding.requires_grad_(False)
        # Checkpoints hold only the DF head (see on_save_checkpoint), so the
        # frozen backbone keys are legitimately absent on load.
        self.strict_loading = False
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    @contextmanager
    def _teacher_eval(self):
        """Temporarily evaluate the shared backbone for an AR teacher call.

        The DF path functionally substitutes weights into this same module, so
        keeping the backbone globally in eval mode would also disable any
        training-mode behavior used by the drafter.
        """
        was_training = self.orthrus.model.training
        self.orthrus.model.eval()
        try:
            yield
        finally:
            self.orthrus.model.train(was_training)

    @contextmanager
    def _frozen_val_rng(self, batch_idx: int):
        """Make stochastic validation inputs repeatable without perturbing training."""
        # An EMPTY device list, not None: with None ``fork_rng`` enumerates every
        # device of the given type and asks each for its RNG state, and
        # ``torch.cpu`` exposes no ``get_rng_state``. The CPU generator is forked
        # regardless, which is the one this needs.
        devices = [] if self.device.type == "cpu" else [self.device]
        with torch.random.fork_rng(devices=devices, device_type=self.device.type):
            torch.manual_seed(int(self.cfg.seed) * 1_000_003 + batch_idx)
            yield

    # --- mechanism passthrough ------------------------------------------------

    def forward(self, *args, **kwargs):
        """Delegate to the adapter: ``forward(ids_or_simplex, mask, use_df=..., s=..., t=...)``."""
        return self.orthrus(*args, **kwargs)

    # --- CFM trajectory (design knob: override to try other schedules) --------

    def sample_prior(self, simplex, attention_mask=None):
        """Sample the prior consumed by one-jump training and decoding.

        The choice decides whether the interpolant ``x_s = (1-s) x0 + s x1``
        conceals the answer or hands it over. Writing ``t*`` for the level at
        which the clean token becomes the argmax of the input, measured at
        ``V = 151936``:

        ============ ======== ==========================================
        prior        ``t*``   share of ``t ~ U[0,1]`` that is ambiguous
        ============ ======== ==========================================
        dirichlet    7.4e-5   0.007%
        discunif     0.500    50%
        gaussian     0.815    81%
        ============ ======== ==========================================

        Under ``dirichlet`` the prior's largest component is about
        ``ln(V)/V ≈ 9e-5``, so a spike of size ``s`` on the clean token
        dominates it almost immediately and every term that reads ``x_s`` is
        solved by copying the input. ``gaussian`` keeps the largest competing
        component near ``sqrt(2 ln V) ≈ 4.9``, so the answer stays buried until
        ``s`` is large; ``discunif`` puts a single competing spike of size
        ``1 - s`` against the clean ``s``, giving the crossing at one half.

        ``gaussian`` leaves the simplex — the interpolant is then a point in
        ``R^V``, which the embedding ``x @ E`` and the transport
        ``x + γ(π - x)`` both accept, and only the model's OUTPUT has to be a
        distribution.

        Scale matters here in a way it does not in the reference method, whose
        input projection is trained. Measured at ``V = 151936`` against a token
        embedding's norm of 1.0, ``x0 @ E`` comes out at 0.004 for
        ``dirichlet`` (270x too small — the frozen trunk sees essentially the
        mean embedding, which is one reason the map behaves as a constant),
        1.00 for ``discunif``, and 392 for an unscaled Gaussian. ``gaussian``
        is therefore emitted at ``1/sqrt(V)``, matching the ``x/sqrt(V)`` the
        reference feeds its projection. ``discunif`` needs no scaling at all
        and is the safer choice against a frozen embedding.
        """
        vocab = simplex.size(-1)
        kind = str(self.cfg.train.get("prior_type", "dirichlet"))
        shape = simplex.shape[:2]
        if kind == "dirichlet":
            x0 = torch.distributions.Dirichlet(
                torch.ones(vocab, device=simplex.device)
            ).sample(shape)
        elif kind == "gaussian":
            # Scaled by 1/sqrt(V), as the reference implementation feeds
            # x/sqrt(V) to its input projection. Without it the embedding
            # x0 @ E has norm ~sqrt(V) times a token's — measured at V=151936,
            # 391 against 1.0 — and a FROZEN trunk has no way to absorb that.
            x0 = torch.randn(
                *shape, vocab, device=simplex.device, dtype=simplex.dtype
            ) / (vocab ** 0.5)
        elif kind == "discunif":
            idx = torch.randint(vocab, shape, device=simplex.device)
            x0 = F.one_hot(idx, vocab).to(simplex.dtype)
        else:
            raise ValueError(
                f"unknown prior_type='{kind}' "
                "(dirichlet | gaussian | discunif)"
            )
        if attention_mask is not None:
            x0 = x0 * attention_mask[..., None].to(x0.dtype)
        return x0

    # --- the loss is yours ----------------------------------------------------

    def _teacher_loss(self, teacher_logits, logits, live, sample_weight=None,
                      position_weight=None):
        """Match the drafter to the frozen AR path, in one of two senses.

        ``train.teacher_target``:

        * ``soft`` — ``KL(sg(p_AR) || π)``. Minimised at ``π = p_AR``, so with an
          attainable target this is the complete objective.
        * ``hard`` — ``CE(argmax p_AR, π)``. Minimised by putting the mode where
          the verifier's mode is, and indifferent to the rest of the mass.
        * ``tv`` — total variation ``½ Σ |p_AR − π|``. Speculative sampling
          accepts a proposal with probability ``Σ min(p, q) = 1 − TV(p, q)``, so
          for sampled decoding this is not a surrogate for the acceptance rate
          but the acceptance rate itself, up to sign.

        Which one matches the metric depends on how the drafter will be
        decoded, because the two verification rules read different things.
        Greedy accepts on ``argmax π == argmax p_AR`` and ignores the rest of
        the distribution; speculative sampling reads all of it. Note that
        neither choice can affect output quality — verification makes the
        emitted text identical to the AR model's under both rules — so this
        only ever trades acceptance in one decoding mode against the other.

        The choice matters because the target is NOT attainable: ``p_AR`` at a
        block position is conditioned on the clean tokens before it, which the
        drafter does not have, so it can only represent a mixture over the
        predecessors it is uncertain about. Under that constraint a forward KL
        spends capacity matching mass that greedy verification never reads, and
        a blurred mixture can score better on it while its argmax sits on a
        different token than the verifier's — lower loss, rejected block.
        Greedy acceptance is exactly ``argmax π == argmax p_AR``, which is what
        the hard target optimises directly.

        Keep ``soft`` for sampled decoding: the coupled-Gumbel scheme accepts
        against the whole proposal distribution, not just its mode.
        """
        mode = str(self.cfg.train.get("teacher_target", "soft"))
        if mode not in ("soft", "hard", "tv"):
            raise ValueError(f"unknown teacher_target='{mode}' (soft | hard | tv)")
        if not live.any():
            return logits.sum() * 0.0
        # Все три режима идут ОДНИМ чанкованным путём. Раньше чанкование было
        # только у `soft`, и `hard` при бумажном пресете стоил 2.0 ГБ, `tv` --
        # 5.1 ГБ на вызов: `logits.float()`, копия, которую cross_entropy
        # заставляет сделать из транспонирования, и две полные fp32-софтмаксы
        # соответственно.
        vocab = logits.size(-1)
        flat_q = logits.reshape(-1, vocab)
        flat_p = teacher_logits.reshape(-1, vocab)
        live_f = live.reshape(-1).to(torch.float32)
        factor = live_f
        if sample_weight is not None:
            # Вес примера СЖИМАЕТ член (расписание (1-t)^p гасит учителя),
            # поэтому в знаменатель не входит.
            factor = factor * sample_weight[:, None].expand_as(live).reshape(
                -1).to(torch.float32)
        if position_weight is not None:
            # Вес позиции ПЕРЕРАСПРЕДЕЛЯЕТ внутри блока и не должен менять
            # масштаб члена целиком, иначе баланс с прочими членами поплывёт
            # вслед за свойством данных. Деление на реализованную массу веса
            # держит масштаб на месте.
            w_f = position_weight.expand_as(live).reshape(-1).to(torch.float32)
            factor = factor * w_f
            denom = (w_f * live_f).sum().clamp_min(1e-6)
        else:
            denom = live_f.sum().clamp_min(1e-6)

        def chunk_term(q_chunk, p_chunk, f_chunk):
            if mode == "hard":
                per_token = F.cross_entropy(
                    q_chunk.float(), p_chunk.argmax(-1), reduction="none",
                )
            elif mode == "tv":
                per_token = 0.5 * (
                    F.softmax(p_chunk.float(), -1) - F.softmax(q_chunk.float(), -1)
                ).abs().sum(-1)
            else:
                log_q = F.log_softmax(q_chunk.float(), -1)
                log_p = F.log_softmax(p_chunk.float(), -1)
                per_token = (log_p.exp() * (log_p - log_q)).sum(-1)
            return (per_token * f_chunk).sum()

        size = self._kl_chunk_rows(vocab)
        total = flat_q.new_zeros((), dtype=torch.float32)
        for start in range(0, flat_q.size(0), size):
            stop = start + size
            total = total + checkpoint(
                chunk_term, flat_q[start:stop], flat_p[start:stop],
                factor[start:stop], use_reentrant=False,
            )
        return total / denom

    def _kl_chunk_rows(self, vocab):
        """Сколько строк считать за раз в чанкованном KL.

        Константа здесь не работает: кусок стоит `строки * V * 4` байт на
        тензор, и при V=151936 значение 4096 означает 2.5 ГБ на тензор, то
        есть весь бумажный пресет (3584 строки) укладывается в ОДИН кусок и
        чанкование не делает ничего. Бюджет задаётся в байтах на тензор и
        переводится в строки по фактическому словарю; `train.kl_chunk`, если
        задан явно, имеет приоритет.
        """
        explicit = self.cfg.train.get("kl_chunk", None)
        if explicit:
            return int(explicit)
        budget = int(self.cfg.train.get("kl_chunk_bytes", 256 * 1024 * 1024))
        return max(256, budget // max(int(vocab), 1) // 4)

    def _assert_finite(self, loss, batch_idx):
        """Проверка конечности лосса КОЛЛЕКТИВНАЯ.

        NaN под bf16 обычно появляется на одном ранге. Если этот ранг выбросит
        исключение в одиночку, остальные останутся ждать в all-reduce
        градиентов, пока не сработает сторож NCCL (порядка получаса), и прогон
        сообщит о падении не там, где оно случилось. Один маленький all_reduce
        на шаг стоит дёшево и делает падение одновременным.
        """
        bad = torch.zeros((), device=loss.device, dtype=torch.float32)
        if not torch.isfinite(loss):
            bad = bad + 1.0
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            bad = trainer.strategy.reduce(bad, reduce_op="sum")
        if bad.item() > 0:
            raise ValueError(f"non-finite loss at step {batch_idx}: {loss}")

    # --- generation: the model generates, start to finish ----------------------
    # src/eval.py drives these: it compares generate() vs ar_generate() and
    # asserts losslessness.

    @staticmethod
    def _jump_schedule(jumps):
        """Normalise a schedule to a list of ``(s, t)`` refinement passes.

        Three accepted forms:

        * ``int n`` — n equal refinement passes over ``linspace(0, 1)``: ``(0, 1/n),
          (1/n, 2/n), ...``. Each refinement pass advances the state a little.
        * ``list of times`` ``[0, u, 1]`` — the same thing written out.
        * ``list of pairs`` ``[(0, 1), (s, 1)]`` — refinement passes that need NOT chain.
          ``(s, 1)`` after ``(0, 1)`` means "draft the whole way, then re-enter
          the family at s carrying that draft", which is a different operation
          from splitting the interval and the only one the self-correction term
          trains: it supervises ``π_{s,1}`` on a state built from the model's
          own completed draft, never on a half-advanced interpolant. A schedule
          of chained refinement passes asks the map questions at pairs whose inputs it was
          not shown.

        Every refinement pass must satisfy ``0 <= s < t <= 1``; the first must start at 0
        and the last must end at 1, so the deployed map is still ``·, 1``.
        """
        # Нормализация к обычным питоновским контейнерам, и НА ЛЮБОЙ глубине.
        # Из конфига сюда приезжает ListConfig, который не является ни list, ни
        # tuple, поэтому проверка на вложенность ниже его не узнавала: пары
        # [[s,t],...] уходили в разбор плоского списка времён, где
        # float(ListConfig) падает. Первая починка снимала только внешнюю
        # обёртку, а вызывающие делают list(...) заранее — тогда внешний объект
        # уже обычный список, а ЭЛЕМЕНТЫ всё ещё ListConfig, и падение
        # повторялось. Проверять надо каждый уровень.
        if OmegaConf.is_config(jumps):
            jumps = OmegaConf.to_container(jumps, resolve=True)
        elif isinstance(jumps, (list, tuple)):
            jumps = [
                OmegaConf.to_container(x, resolve=True)
                if OmegaConf.is_config(x) else x
                for x in jumps
            ]
        if isinstance(jumps, int):
            times = torch.linspace(0, 1, jumps + 1).tolist()
            passes = list(zip(times[:-1], times[1:]))
        else:
            items = list(jumps)
            if items and isinstance(items[0], (list, tuple)):
                passes = [(float(a), float(b)) for a, b in items]
            else:
                times = [float(x) for x in items]
                passes = list(zip(times[:-1], times[1:]))
        if not passes:
            raise ValueError("jump schedule must contain at least one refinement pass")
        if any(not 0.0 <= s < t <= 1.0 for s, t in passes):
            raise ValueError(f"every refinement pass must satisfy 0 <= s < t <= 1, got {passes}")
        if passes[0][0] != 0.0 or passes[-1][1] != 1.0:
            raise ValueError(
                f"a schedule must start at s=0 and finish at t=1, got {passes}"
            )
        return passes

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

    def _generation_device(self):
        """Device of the token embedding stage under HF device-map dispatch.

        ``LightningModule.device`` only tracks explicit ``module.to(...)``
        calls. Evaluation can instead place the wrapped backbone with a
        Hugging Face device map, leaving ``self.device`` at CPU even though
        the embedding and generated tensors live on CUDA.
        """
        embedding = self.orthrus.model.get_input_embeddings()
        hook = getattr(embedding, "_hf_hook", None)
        device = getattr(hook, "execution_device", None)
        device = torch.device(device) if device is not None else embedding.weight.device
        if device.type == "meta":
            raise RuntimeError(
                "generation cannot target a meta-device embedding; choose an "
                "evaluation device_map that materializes the input embedding"
            )
        return device

    def _encode(self, text, input_ids, **tokenizer_kwargs):
        if (text is None) == (input_ids is None):
            raise ValueError("pass exactly one of text / input_ids")
        if text is not None:
            enc = self.df_processor(text, return_simplex=False, **tokenizer_kwargs)
            input_ids = enc["input_ids"]
        input_ids = input_ids.to(self._generation_device())
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

        ``carry`` — ``(q_prev, n_accepted)`` from the cycle that just ended, the
        decode-side half of ``onpolicy_kl_weight``. Today every cycle throws its
        rejected tail away and starts from pure noise, which is the one place
        where a diffusion drafter is strictly worse informed than it needs to
        be: it already guessed those tokens, and the verifier already told it
        where the guess went wrong.

        The shift is ``n_accepted + 1``, not one: the verifier consumed the
        accepted prefix AND replaced the first mismatch with its own token, so
        the new block's position ``j`` is the old block's ``n_accepted + 1 + j``.
        Carried positions enter at ``decode.onpolicy_s`` and fresh ones at 0 --
        different times in the same block, which is why this needs per-position
        conditioning. It costs no forward: the state is already in hand.

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

        ``block_size`` is the total parallel width: one clean anchor plus
        ``K-1`` fresh positions. Returns draft tensors with shape
        ``[1, K-1]``.
        """
        vocab = self.df_processor.vocab_size
        device = self._generation_device()
        drafted = block_size - 1
        if drafted <= 0:
            raise ValueError("block_size must be at least 2 (anchor + one draft)")
        decode_cfg = self.cfg.get("decode", {}) if hasattr(self.cfg, "get") else {}
        # Draw through sample_prior so the decode entry state is the SAME
        # distribution the model was trained on. Hardcoding a family here would
        # mismatch train and inference on the one input the deployed map ever
        # reads, and no metric produced by such a run would mean anything.
        x = self.sample_prior(
            torch.zeros(1, drafted, vocab, device=device)
        )
        times = list(times)
        if bool(decode_cfg.get("fixed_prior", False)):
            # Greedy verification accepts on an argmax match, a criterion with
            # no randomness in it, so redrawing the prior each cycle only adds
            # variance to that one input. Freeze it instead — deterministically
            # per cycle, but still a sample from the training prior rather than
            # its mean, which for a one-hot prior is the uniform point and for
            # a Dirichlet prior embeds to the vocabulary mean. Sampled decoding
            # keeps its randomness from the proposal draw and the coupled
            # Gumbel noise, neither of which comes from here.
            # На CPU список устройств ПУСТОЙ, а не None: с None torch пытается
            # взять torch.cpu.get_rng_state, которого не существует, и
            # decode.fixed_prior=true падает на процессорном фолбэке.
            with torch.random.fork_rng(
                devices=[device] if device.type != "cpu" else [],
                device_type=device.type,
            ):
                torch.manual_seed(int(self.cfg.get("seed", 0)))
                x = self.sample_prior(torch.zeros(1, drafted, vocab, device=device))
        anchor = None
        if anchor_token is not None:
            anchor = F.one_hot(
                anchor_token.to(device).view(1, 1), vocab
            ).to(x.dtype)
            x = torch.cat([anchor, x], dim=1)
        mask = torch.ones(
            1,
            cache.get_seq_length() + x.size(1),
            dtype=torch.long,
            device=x.device,
        )
        previous_t = None
        for pass_idx, (s_i, t_i) in enumerate(times):
            if previous_t is not None and abs(s_i - previous_t) > 1e-9:
                # This refinement pass does not continue the previous one: it RE-ENTERS the
                # family at s_i. The state it should read is the one training
                # built at that time — a fresh prior draw mixed with the draft
                # in hand — not the transported point left by the previous refinement pass,
                # which sits at a different time and would put the map at a pair
                # it was never shown. Rebuilding it here is what makes a
                # schedule like [(0,1), (0.5,1)] the operation it reads as.
                # Розыгрыш рестарта тоже обязан подчиняться fixed_prior. Он
                # лежал вне этого блока, поэтому при n > 1 шум второго шага уточнения брался
                # из глобального RNG, который замер нигде не сеет: конфигурации не были
                # спарены по этой оси вообще, а у маскирующего драфтера её нет.
                if bool(decode_cfg.get("fixed_prior", False)):
                    with torch.random.fork_rng(
                        devices=[device] if device.type != "cpu" else [],
                        device_type=device.type,
                    ):
                        torch.manual_seed(int(self.cfg.get("seed", 0)) * 7919 + pass_idx)
                        fresh = self.sample_prior(x)
                else:
                    fresh = self.sample_prior(x)
                x = (1.0 - s_i) * fresh + s_i * x
                if anchor is not None:
                    x = torch.cat([anchor, x[:, 1:]], dim=1)
            previous_t = t_i
            # One scalar time per refinement pass. A per-position clock existed here for the
            # carried-tail state; that state is gone (bucket/README.md), and with
            # it the only caller. Announcing per-position times when the block is
            # NOT mixed is the conditioning that cost 1.16 TPF when it slipped in.
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

        Every cycle: the flow map drafts ``block_size - 1`` fresh tokens in
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
                    n_forwards, acceptance=acceptance, prefill_tokens=1,
                )

        while len(emitted) < max_new_tokens:
            draft_ids, q = self._draft_block(
                cache, block_size, times,
                sample=temperature > 0 and not coupled, anchor_token=pending,
            )
            n_forwards += len(times)
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
            mask = torch.ones(
                1,
                committed + verify_in.size(1),
                dtype=torch.long,
                device=verify_in.device,
            )
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
                              acceptance=acceptance, prefill_tokens=int(bool(emitted)))

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
            mask = torch.ones(
                1,
                cache.get_seq_length() + 1,
                dtype=torch.long,
                device=token.device,
            )
            out = self.orthrus(token.view(1, 1), mask, past_key_values=cache)
            n_forwards += 1

        return self._finalize(input_ids, emitted, max_new_tokens, eos_token_id, start, n_forwards)

    def _finalize(self, input_ids, emitted, max_new_tokens, eos_token_id, start, n_forwards,
                  acceptance=None, cycle_forwards=None, prefill_tokens=0):
        produced = len(emitted)
        emitted = emitted[:max_new_tokens]
        if eos_token_id is not None and eos_token_id in emitted:
            emitted = emitted[: emitted.index(eos_token_id) + 1]
        result = {
            "sequences": torch.cat(
                [input_ids, torch.tensor([emitted], device=input_ids.device)], dim=1
            ),
            "new_tokens": emitted,
            "n_forwards": n_forwards,
            # End-to-end n_forwards charges the run for the prefill and for the
            # last cycle in full even though its overflow past max_new_tokens is
            # discarded above, so tokens/n_forwards depends on how long the
            # generation was asked to be — two systems are only comparable
            # through it at an identical max_new_tokens. These two report the
            # steady-state rate instead: every token the cycles actually
            # produced, over the forwards those cycles actually cost.
            # `prefill_tokens` — то, что вышло из ПРЕФИЛЛА, а не из циклов:
            # первый токен материализуется прямо из распределения префилла и не
            # стоит ни одного прохода цикла. В числителе установившейся скорости
            # ему не место — иначе она завышена ровно на `1/produced`, то есть
            # на 3.1% при 32 новых токенах и на 1.6% при 64, и перестаёт быть
            # длинно-независимой, чем и объявлена. Замерено: смещение падает
            # как 1/produced на длинах 32, 64 и 128.
            "produced_tokens": produced - prefill_tokens,
            "cycle_forwards": n_forwards - 1 if cycle_forwards is None else cycle_forwards,
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
        s = torch.as_tensor(s, dtype=pi.dtype, device=pi.device)
        t = torch.as_tensor(t, dtype=pi.dtype, device=pi.device)
        # gamma follows the shape of the times: one per sequence broadcasts over
        # the block as before; one per POSITION gives each position its own
        # transport, which is what a block of mixed stages needs -- a confirmed
        # token at t = 1 must be left where it is while a fresh slot moves the
        # whole way.
        if s.dim() <= 1 and s.numel() <= pi.size(0):
            s = s.reshape(-1, 1, 1)
            t = t.reshape(-1, 1, 1)
        else:
            s = s.reshape(pi.size(0), -1, 1)
            t = t.reshape(pi.size(0), -1, 1)
        gamma = (t - s) / (1.0 - s).clamp(min=float(self.cfg.train.get("gamma_clamp", 1e-4)))
        return x_s + gamma * (pi - x_s)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError(
            "variant='flowdraft' trains the whole sequence, and its whole loss "
            "was endpoint + EC + TD, which never carried weight in any preset and "
            "now live in bucket/cfm_terms. Train flowdraft_block_wise or orthrus."
        )

    def on_validation_epoch_start(self):
        # Decode metrics are intentionally sampled from several consecutive
        # validation batches. ``data.batch_size=1`` is required by the paper
        # recipe, so restricting decoding to batch zero would otherwise turn
        # any requested sample count into a single prompt.
        # Счётчик делится между рангами: он инициализируется НА КАЖДОМ ранге,
        # и без деления `val_decode_prompts: 16` означал бы 128 промптов на
        # восьми GPU. Тогда val/tpf, посчитанный на одной машине, не сравним с
        # посчитанным на другой -- это была бы другая величина, а не та же с
        # шумом.
        world = max(1, int(getattr(self.trainer, "world_size", 1) or 1))
        self._val_decode_remaining = (
            0 if self.trainer.sanity_checking
            else max(1, int(self.cfg.train.get("val_decode_prompts", 0)) // world)
            if self.cfg.train.get("val_decode_prompts", 0) else 0
        )
        self._val_decode_accs = []
        self._val_decode_tpfs = []
        drafted = max(int(self.cfg.train.get("block_size", 8)) - 1, 0)
        max_cycles = max(int(self.cfg.train.get("val_decode_max_new", 32)), 0)
        # Pooled real-decode statistics. Position j records how many actual
        # speculative cycles accepted at least j drafts. Cycle j records the
        # accepted-draft count for prompts that reached that generation cycle.
        self._val_decode_position_hits = torch.zeros(drafted, dtype=torch.float64)
        self._val_decode_cycle_sums = torch.zeros(max_cycles, dtype=torch.float64)
        self._val_decode_cycle_counts = torch.zeros(max_cycles, dtype=torch.float64)
        self._val_decode_cycle_count = 0
        self._val_mixed_done = False

    def on_validation_epoch_end(self):
        requested = self.cfg.train.get("val_decode_prompts", 0)
        if self.trainer.sanity_checking or requested <= 0:
            return

        # Reduce sums and counts instead of averaging rank-local means. This
        # remains correct when the final validation shard is uneven, and all
        # ranks participate even if one rank had no usable prompt.
        accum = self._accum_dtype(self.device)
        position_hits = self._val_decode_position_hits.to(self.device, accum)
        cycle_sums = self._val_decode_cycle_sums.to(self.device, accum)
        cycle_counts = self._val_decode_cycle_counts.to(self.device, accum)
        stats = torch.cat(
            [
                torch.tensor(
                    [
                        sum(self._val_decode_tpfs),
                        len(self._val_decode_tpfs),
                        sum(self._val_decode_accs),
                        len(self._val_decode_accs),
                        self._val_decode_cycle_count,
                    ],
                    dtype=accum,
                    device=self.device,
                ),
                position_hits,
                cycle_sums,
                cycle_counts,
            ]
        )
        stats = self.trainer.strategy.reduce(stats, reduce_op="sum")
        if stats[1].item() == 0:
            raise RuntimeError(
                "val/tpf selection requested, but validation decoded no usable "
                "prompts; increase the validation data/limit or inspect prompt lengths"
            )
        if stats[3].item() > 0:
            self.log(
                "val/acceptance_decode",
                stats[2] / stats[3],
                # ``stats`` is already an all-rank sum, so every rank logs
                # the same global ratio. Lightning's mean synchronization is
                # idempotent here and keeps distributed checkpoint monitors
                # warning-free.
                sync_dist=True,
            )
        drafted = position_hits.numel()
        max_cycles = cycle_sums.numel()
        cycle_count = stats[4]
        cursor = 5
        global_position_hits = stats[cursor : cursor + drafted]
        cursor += drafted
        global_cycle_sums = stats[cursor : cursor + max_cycles]
        cursor += max_cycles
        global_cycle_counts = stats[cursor : cursor + max_cycles]

        decode_metrics = {}
        if cycle_count.item() > 0:
            position_acceptance = global_position_hits / cycle_count
            decode_metrics.update(
                {
                    f"val/decode/acceptance_pos_{position + 1:02d}": value
                    for position, value in enumerate(position_acceptance)
                }
            )
            # Tail-sum identity: the expected accepted-draft count equals the
            # sum of P(n_accepted >= j) over all draft positions.
            decode_metrics["val/decode/accepted_mean"] = position_acceptance.sum()
        decode_metrics.update(
            {
                f"val/decode/accepted_cycle_{cycle + 1:02d}": (
                    global_cycle_sums[cycle] / global_cycle_counts[cycle]
                )
                for cycle in range(max_cycles)
                if global_cycle_counts[cycle].item() > 0
            }
        )
        if decode_metrics:
            # Every value was explicitly reduced above and is identical on
            # each rank; another Lightning synchronization is unnecessary.
            self.log_dict(decode_metrics, sync_dist=False)
        self.log(
            "val/tpf",
            stats[0] / stats[1],
            prog_bar=True,
            sync_dist=True,
        )

    @staticmethod
    def _accum_dtype(device=None):
        """Widest accumulator this device actually has.

        Sums of counts and of per-token losses are accumulated in float64 for
        exactness across a whole validation epoch. MPS has no float64 at all --
        it raises rather than downcasting -- and every such call sat on a code
        path that only the masked baseline reaches, so the baseline was simply
        untrainable on Apple silicon while looking like a stalled process.
        """
        return torch.float32 if device is not None and device.type == "mps" else torch.float64

    @staticmethod
    def _decode_acceptance_parts(acceptance, drafted, max_cycles):
        """Sufficient statistics for on-policy positional/cycle acceptance.

        ``acceptance[c]`` is the number of drafts accepted in real generation
        cycle ``c``. The positional tail counts satisfy
        ``sum_j P(acceptance >= j) == mean(acceptance)``.
        """
        accepted = torch.as_tensor(acceptance, dtype=torch.long)
        position_hits = torch.zeros(drafted, dtype=torch.float64)
        cycle_sums = torch.zeros(max_cycles, dtype=torch.float64)
        cycle_counts = torch.zeros(max_cycles, dtype=torch.float64)
        if accepted.numel() == 0:
            return position_hits, cycle_sums, cycle_counts, 0
        if (accepted < 0).any() or (accepted > drafted).any():
            raise ValueError(
                f"decode acceptance must be between 0 and {drafted}, "
                f"got {accepted.tolist()}"
            )
        if drafted:
            positions = torch.arange(1, drafted + 1)
            position_hits = (accepted[:, None] >= positions[None]).sum(0).to(torch.float64)
        observed_cycles = min(accepted.numel(), max_cycles)
        cycle_sums[:observed_cycles] = accepted[:observed_cycles].to(torch.float64)
        cycle_counts[:observed_cycles] = 1.0
        return position_hits, cycle_sums, cycle_counts, int(accepted.numel())

    def _mixed_val_prompts(self):
        """Fixed held-out prompts drawn evenly from several benchmark datasets.

        Validation used to decode the first samples of the TRAINING stream, so
        checkpoint selection was made on the training distribution. With
        ``train.val_decode_datasets`` set, the decode instead runs on an even
        mix of the named benchmarks -- the distributions the drafter is judged
        on. Built once and cached: these are streaming datasets and rebuilding
        them every validation would dominate the epoch.
        """
        cached = getattr(self, "_val_prompt_cache", None)
        if cached is not None:
            return cached
        names = list(self.cfg.train.get("val_decode_datasets", []) or [])
        if not names:
            self._val_prompt_cache = []
            return []
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict
        from src.eval import dataset_prompts
        import os

        total = int(self.cfg.train.get("val_decode_prompts", 0))
        per = max(1, total // len(names))
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "configs")
        prompts = []
        for name in names:
            # `train.py:main` уже стоит под @hydra.main, то есть GlobalHydra
            # инициализирована, и вложенная инициализация валится с
            # "GlobalHydra is already initialized". Обёртка try ниже ловит
            # только загрузку датасета, так что это падало бы наружу из
            # validation_step на ПЕРВОЙ валидации. Под запущенной Hydra
            # достаточно самого compose: путь поиска уже включает эти конфиги.
            if GlobalHydra.instance().is_initialized():
                sub = compose(config_name="eval", overrides=[f"data={name}"])
            else:
                with initialize_config_dir(config_dir=root, version_base=None):
                    sub = compose(config_name="eval", overrides=[f"data={name}"])
            with open_dict(sub):
                sub.model = self.cfg.model
                sub.decode.n_prompts = per
                sub.decode.prompt_len = 48
            try:
                prompts += [(name, p) for _, _, p in dataset_prompts(self, sub)]
            except Exception as error:  # a dataset that will not load must not
                logger.warning(f"validation dataset {name!r} skipped: {error}")
        self._val_prompt_cache = prompts
        logger.info(
            f"validation decode uses {len(prompts)} prompts from "
            f"{len(names)} datasets ({per} each)"
        )
        return prompts

    @torch.no_grad()
    def _maybe_decode_val(self, batch, batch_idx):
        """The REAL target metrics as validation curves: run the lossless
        decode loop (single-jump — the headline configuration) on a few val
        prompts and log ``val/acceptance_decode``, ``val/decode/*``, and
        ``val/tpf``. This is what training should improve to beat the baseline,
        and what the checkpoint monitor tracks; ``train.val_decode_prompts=0``
        disables.
        """
        remaining = getattr(self, "_val_decode_remaining", 0)
        if remaining <= 0:
            return
        mixed = self._mixed_val_prompts()
        if mixed:
            # Смешанный набор фиксирован и не зависит от батча, поэтому он
            # проходится целиком на ПЕРВОМ валидационном батче, а дальше
            # валидация ничего не декодирует.
            if getattr(self, "_val_mixed_done", False):
                return
            self._val_mixed_done = True
            accs, tpfs, decoded = [], [], 0
            max_new = self.cfg.train.get("val_decode_max_new", 32)
            block = self.cfg.train.get("block_size", 8)
            vj = self.cfg.train.get("val_decode_jumps", 1)
            for _, ids in mixed:
                out = self.generate(input_ids=ids, block_size=block, jumps=vj,
                                    max_new_tokens=max_new)
                if out["acceptance"]:
                    accs.append(sum(out["acceptance"]) / len(out["acceptance"]))
                parts = self._decode_acceptance_parts(
                    out["acceptance"],
                    drafted=self._val_decode_position_hits.numel(),
                    max_cycles=self._val_decode_cycle_sums.numel())
                self._val_decode_position_hits += parts[0]
                self._val_decode_cycle_sums += parts[1]
                self._val_decode_cycle_counts += parts[2]
                self._val_decode_cycle_count += parts[3]
                tpfs.append(len(out["new_tokens"]) / out["n_forwards"])
                decoded += 1
            self._val_decode_remaining = 0
            self._val_decode_accs.extend(accs)
            self._val_decode_tpfs.extend(tpfs)
            return
        max_new = self.cfg.train.get("val_decode_max_new", 32)
        block = self.cfg.train.get("block_size", 8)
        # The schedule validation decodes with, and therefore the schedule
        # checkpoint selection and early stopping see. Hardcoding one jump means
        # a run aiming at multi-step would select on the metric it is not aiming
        # at: val/tpf is the monitor, and a model better at two jumps can lose
        # the selection to one that is better at one.
        val_jumps = self.cfg.train.get("val_decode_jumps", 1)
        accs, tpfs = [], []
        decoded = 0
        for i in range(min(remaining, batch["input_ids"].size(0))):
            live = int(batch["attention_mask"][i].sum())
            plen = min(max(live // 2, 2), 32)
            if live < plen + 2:
                continue
            out = self.generate(
                input_ids=batch["input_ids"][i : i + 1, :plen],
                block_size=block, jumps=val_jumps, max_new_tokens=max_new,
            )
            if out["acceptance"]:
                accs.append(sum(out["acceptance"]) / len(out["acceptance"]))
            position_hits, cycle_sums, cycle_counts, cycle_count = (
                self._decode_acceptance_parts(
                    out["acceptance"],
                    drafted=self._val_decode_position_hits.numel(),
                    max_cycles=self._val_decode_cycle_sums.numel(),
                )
            )
            self._val_decode_position_hits += position_hits
            self._val_decode_cycle_sums += cycle_sums
            self._val_decode_cycle_counts += cycle_counts
            self._val_decode_cycle_count += cycle_count
            tpfs.append(len(out["new_tokens"]) / out["n_forwards"])
            decoded += 1
        self._val_decode_remaining -= decoded
        self._val_decode_accs.extend(accs)
        self._val_decode_tpfs.extend(tpfs)

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

    def on_train_start(self):
        self._campaign_started_at = time.perf_counter()

    def _campaign_metadata(self):
        elapsed = float(getattr(self, "_campaign_elapsed_before", 0.0))
        started = getattr(self, "_campaign_started_at", None)
        if started is not None:
            elapsed += time.perf_counter() - started
        world_size = int(getattr(self.trainer, "world_size", 1))
        return {
            "elapsed_seconds": elapsed,
            "device_count": world_size,
            "device_hours": elapsed * world_size / 3600.0,
        }

    def on_save_checkpoint(self, checkpoint):
        # Keep only the FP32 DF head; build_model restores the much larger
        # frozen backbone from Hugging Face on load.
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            key: value for key, value in checkpoint["state_dict"].items() if key in trainable
        }
        checkpoint["campaign_metadata"] = self._campaign_metadata()

    def on_load_checkpoint(self, checkpoint):
        """Validate DF-only state and migrate pre-mask-freeze optimizer state."""
        self._campaign_elapsed_before = float(
            checkpoint.get("campaign_metadata", {}).get("elapsed_seconds", 0.0)
        )
        state = checkpoint["state_dict"]
        legacy_mask = "orthrus.mask_embedding"
        if legacy_mask in state and not self.orthrus.mask_embedding.requires_grad:
            # vcstk checkpoints optimized an unused FlowDraft mask parameter.
            # Remove that tensor and its Adam slot while preserving every
            # other parameter, scheduler value, and global step.
            state.pop(legacy_mask)
            current_count = len(list(self.orthrus.df_parameters()))
            mask_position = len(self.orthrus.df_weights)
            for optimizer_state in checkpoint.get("optimizer_states", []):
                groups = optimizer_state.get("param_groups", [])
                if len(groups) != 1 or len(groups[0].get("params", [])) != current_count + 1:
                    raise RuntimeError(
                        "cannot migrate legacy FlowDraft optimizer state: expected "
                        "one parameter group containing exactly one obsolete mask parameter; "
                        "use the checkpoint as a weights-only warm start instead"
                    )
                parameter_ids = groups[0]["params"]
                removed_id = parameter_ids.pop(mask_position)
                optimizer_state.get("state", {}).pop(removed_id, None)
            logger.warning(
                "migrated legacy FlowDraft checkpoint by removing the unused mask "
                "parameter from model and optimizer state"
            )

        from src.models.factory import validate_df_state

        validate_df_state(self, state, "resume checkpoint")
