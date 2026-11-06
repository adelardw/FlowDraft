"""Flow-matching velocity regression plus FlowDraft's weighted multistep KL."""

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from src.models.flowdraft import FlowDraft
from src.models.flowdraft_block_wise import FlowDraftBlockWise


class FlowDraftFM(FlowDraftBlockWise):
    """Gaussian-to-token flow with a time-weighted velocity regression loss.

    The endpoint head parametrises v(x,t) = (softmax(logits) - x)/(1-t).
    On x_t=(1-t)x_0+t*x_1, weighting squared velocity error by (1-t)^2
    gives exactly the squared endpoint error. Compute that equivalent form
    directly to avoid subtracting large velocities near t=1.

    FM endpoints come from the supplied token blocks. A separate Euler
    trajectory receives the original weighted multistep KL on q=x+(1-t)*v.
    Each refinement uses the verifier's response to the previous draft;
    states are detached between steps. All drafts are verified together
    in one frozen AR call after the context pass. There is no first-pass KL
    or teacher-velocity MSE. This auxiliary KL is not a proof that the joint
    flow distribution matches AR, or that its gradients agree with FM.
    """

    def __init__(self, cfg, **kwargs):
        if cfg.get("trainer") is not None:
            horizon = cfg.trainer.get("max_steps")
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
                raise ValueError("choose a positive trainer.max_steps for this experiment before training")
        super().__init__(cfg, **kwargs)
        self.noise_std = float(cfg.train.get("fm_noise_std", 1.0))
        if not math.isfinite(self.noise_std) or self.noise_std <= 0:
            raise ValueError("fm_noise_std must be finite and positive")
        if cfg.train.get("prior_type") != "gaussian":
            raise ValueError("flowdraft_fm requires prior_type=gaussian")
        if cfg.train.get("fm_loss") != "weighted_velocity_mse":
            raise ValueError("flowdraft_fm implements fm_loss=weighted_velocity_mse")
        if cfg.train.get("teacher_source") not in ("corpus", "fixed_ar_bank") or cfg.train.get("teacher_target") != "soft":
            raise ValueError("flowdraft_fm uses supplied token blocks for FM and soft AR targets for multistep KL")
        if cfg.train.get("teacher_source") == "fixed_ar_bank" and type(self) is FlowDraftFM:
            raise ValueError("fixed_ar_bank requires the flowdraft_fm_bank variant")
        if int(cfg.train.block_size) < 2 or int(cfg.train.anchors_per_sequence) < 1:
            raise ValueError("flowdraft_fm needs block_size>=2 and at least one anchor")
        if any(p.requires_grad for p in self.orthrus.model.parameters()):
            raise ValueError("the AR teacher must remain frozen")
        if cfg.model.backbone.get("compile_ar", False):
            raise ValueError("flowdraft_fm currently uses the eager frozen teacher")
        self.train_step_choices = tuple(cfg.train.get("fm_train_steps", []))
        if not self.train_step_choices or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in self.train_step_choices
        ):
            raise ValueError("fm_train_steps must be a nonempty list of positive integers")
        if any(float(cfg.train.get(key, 0)) != 0 for key in
               ("verify_kl_weight", "verify_velocity_weight", "selfcorrect_velocity_weight")):
            raise ValueError("flowdraft_fm uses only fm_weight and selfcorrect_kl_weight")
        weights = [float(cfg.train.get(key, 0)) for key in
                   ("fm_weight", "selfcorrect_kl_weight")]
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError("loss weights must be finite, nonnegative and not all zero")
        tail = float(cfg.train.get("selfcorrect_tail_weight", 0.5))
        if not 0 <= tail <= 1:
            raise ValueError("selfcorrect_tail_weight must lie in [0, 1]")

    def sample_prior(self, like, attention_mask=None):
        # This is the state prior, with no vocabulary-dependent rescaling.
        x = torch.randn(like.shape, device=like.device, dtype=torch.float32) * self.noise_std
        if attention_mask is not None:
            x = x * attention_mask[..., None]
        return x

    @staticmethod
    def _position_times(t, x):
        t = torch.as_tensor(t, device=x.device, dtype=torch.float32)
        if t.ndim <= 1:
            t = t.reshape(-1, 1)
        if t.ndim != 2:
            raise ValueError("times must be scalar, [B], or [B, positions]")
        return t.expand(x.shape[:2])

    @staticmethod
    def velocity(q, x, t):
        t = FlowDraftFM._position_times(t, x)
        if not bool(((t >= 0) & (t < 1)).all()):
            raise ValueError("velocity is evaluated at 0 <= t < 1")
        return (q.float() - x.float()) / (1 - t[..., None])

    def _field_logits(self, x, t, anchor, ctx_mask, cache, df_kwargs):
        t = self._position_times(t, x)
        # Rescale only the network input, never the ODE state or its prior.
        # The expected squared norm of an interpolated slot is this scale^2.
        scale = ((1 - t).square() * x.size(-1) * self.noise_std**2 + t.square()).sqrt()
        return self._df_forward(
            x / scale[..., None], anchor, ctx_mask, cache, t, t, df_kwargs
        )

    @staticmethod
    def endpoint_squared_loss(logits, target_ids, live):
        """Mean over live positions, sum over vocabulary; equivalent weighted FM."""
        vocab = logits.size(-1)
        flat = logits.reshape(-1, vocab)
        targets, weights = target_ids.flatten(), live.flatten().float()

        def term(z, y, w):
            q = z.float().softmax(-1)
            # ||q - one_hot(y)||^2, without allocating another dense one-hot.
            error = q.square().sum(-1) - 2 * q.gather(-1, y[:, None]).squeeze(-1) + 1
            return (error * w).sum()

        total = flat.new_zeros((), dtype=torch.float32)
        for start in range(0, flat.size(0), 64):
            args = (flat[start:start + 64], targets[start:start + 64], weights[start:start + 64])
            total = total + (checkpoint(term, *args, use_reentrant=False)
                             if torch.is_grad_enabled() else term(*args))
        return total / weights.sum().clamp_min(1)

    def _prepare_fm(self, batch):
        # Use the same complete-window geometry for one and multiple anchors.
        # One AR pass provides the shared frozen KV cache.
        return self._prepare_blocks(batch)

    def _sample_train_steps(self, device):
        index = int(torch.randint(len(self.train_step_choices), (), device=device))
        return self.train_step_choices[index]

    @staticmethod
    def _euler_update(x, q, s, t):
        return q if t == 1.0 else torch.lerp(x, q, (t - s) / (1 - s))

    def _trajectory_logits(self, x, steps, anchor, ctx_mask, cache, geometry):
        predictions = []
        for s, t in self._jump_schedule(steps):
            logits = self._field_logits(x, s, anchor, ctx_mask, cache, geometry)
            predictions.append(logits)
            # Local supervision at every pass, with no gradient through the
            # preceding draft or state, as in the existing multistep loss.
            x = self._euler_update(x, logits.detach().float().softmax(-1), s, t).detach()
        return predictions

    def _trajectory_teacher_logits(self, predictions, anchor_ids, ctx_mask, cache, geometry):
        """Verify all already-produced drafts in one causally isolated AR call."""
        count = len(predictions)
        packed_drafts = torch.cat([z.detach().argmax(-1) for z in predictions], dim=1)
        packed_geometry = {
            **geometry,
            "position_ids": geometry["position_ids"].repeat(1, count),
            "causal_limit": geometry["causal_limit"].repeat(1, count),
        }
        logits = self._verifier_response(
            packed_drafts, anchor_ids.repeat(1, count), ctx_mask, cache, packed_geometry
        )
        return logits.reshape(logits.size(0), count, predictions[0].size(1), -1).unbind(1)

    def _refinement_weights(self, accepted, known, live, anchors):
        tail = float(self.cfg.train.get("selfcorrect_tail_weight", 0.5))
        weights = torch.where(known & ~accepted, 1.0, tail).float()
        survival = self._survival_weights(live.size(1) // anchors, anchors, weights)
        if survival is not None:
            weights = weights * survival
        return weights / weights.mean().clamp_min(1e-6)

    def _multistep_term(self, live, predictions, teacher_predictions, anchors):
        refinements = []
        final_accepted = None
        for k, (logits, teacher) in enumerate(zip(predictions, teacher_predictions)):
            accepted, known = self._greedy_verdict(
                logits.detach().argmax(-1), teacher.argmax(-1), live, anchors
            )
            if k + 1 < len(predictions):
                refinements.append(self._teacher_loss(
                    teacher.detach(), predictions[k + 1], live,
                    position_weight=self._refinement_weights(accepted, known, live, anchors),
                ))
            final_accepted = accepted
        multistep = torch.stack(refinements).mean() if refinements else predictions[0].sum() * 0.0
        accepted_mean = final_accepted.reshape(live.size(0), anchors, -1).sum(-1).float().mean()
        return multistep, accepted_mean

    def _fm_step(self, batch):
        prepared = self._prepare_fm(batch)
        if prepared is None:
            # Keep the graph and subsequent collectives identical on empty ranks.
            zero = sum(p.reshape(-1)[0] for p in self.orthrus.df_parameters()) * 0.0
            return zero, {name: zero for name in
                          ("fm", "selfcorrect_kl", "accepted", "euler_steps")}
        _, targets, mask, live, cache, anchor, anchor_ids, geometry = prepared
        live = live.bool()
        x1 = self.df_processor.to_simplex(targets)
        x0 = self.sample_prior(x1, live)
        times = torch.rand(anchor.shape[:2], device=x1.device)
        times = times.repeat_interleave(int(self.cfg.train.block_size) - 1, dim=1)
        x = (1 - times[..., None]) * x0 + times[..., None] * x1
        logits = self._field_logits(x, times, anchor, mask, cache, geometry)
        fm = self.endpoint_squared_loss(logits, targets, live)

        steps = self._sample_train_steps(x1.device)
        predictions = self._trajectory_logits(
            self.sample_prior(x1, live), steps, anchor, mask, cache, geometry
        )
        teachers = self._trajectory_teacher_logits(predictions, anchor_ids, mask, cache, geometry)
        multistep, accepted = self._multistep_term(
            live, predictions, teachers, anchor.size(1)
        )
        loss = (float(self.cfg.train.fm_weight) * fm
                + float(self.cfg.train.selfcorrect_kl_weight) * multistep)
        return loss, {"fm": fm, "selfcorrect_kl": multistep,
                      "accepted": accepted, "euler_steps": fm.new_tensor(steps)}

    def training_step(self, batch, batch_idx):
        loss, terms = self._fm_step(batch)
        self._assert_finite(loss, batch_idx)
        self.log_dict({f"loss/{name}": value for name, value in terms.items()},
                      on_step=True, on_epoch=False, sync_dist=True,
                      batch_size=batch["input_ids"].size(0))
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False,
                 sync_dist=True, batch_size=batch["input_ids"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        with self._frozen_val_rng(batch_idx):
            loss, terms = self._fm_step(batch)
            self.log_dict({f"val/loss/{name}": value for name, value in terms.items()},
                          on_step=False, on_epoch=True, sync_dist=True,
                          batch_size=batch["input_ids"].size(0))
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=batch["input_ids"].size(0))
            self._maybe_decode_val(batch, batch_idx)
        return loss

    @staticmethod
    def _jump_schedule(jumps):
        if isinstance(jumps, int) and (isinstance(jumps, bool) or jumps < 1):
            raise ValueError("Euler step count must be a positive integer")
        schedule = FlowDraft._jump_schedule(jumps)
        if any(abs(left[1] - right[0]) > 1e-9 for left, right in zip(schedule, schedule[1:])):
            raise ValueError("flowdraft_fm requires consecutive Euler steps; restarts are not supported")
        return schedule

    def predict(self, *args, **kwargs):
        raise NotImplementedError("flowdraft_fm uses its block ODE sampler, not endpoint restart maps")

    @torch.no_grad()
    def _draft_block(self, cache, block_size, times, sample=False, anchor_token=None, carry=None, trace=None):
        if anchor_token is None or carry is not None:
            raise ValueError("flowdraft_fm requires a clean anchor and no carried draft")
        if block_size < 2:
            raise ValueError("block_size must include an anchor and at least one fresh position")
        times = self._jump_schedule(times)
        device, vocab = self._generation_device(), self.df_processor.vocab_size
        like = torch.empty(1, block_size - 1, vocab, device=device)
        if self.cfg.get("decode", {}).get("fixed_prior", False):
            with self._seeded_rng(int(self.cfg.get("seed", 0))):
                x = self.sample_prior(like)
        else:
            x = self.sample_prior(like)
        anchor = F.one_hot(anchor_token.to(device).reshape(1, 1), vocab).float()
        mask = torch.ones(1, cache.get_seq_length() + block_size, device=device, dtype=torch.long)
        for s, t in times:
            q = self._field_logits(x, s, anchor, mask, cache, {}).float().softmax(-1)
            if trace is not None:
                trace.append({'t_start': s, 't_end': t,
                              'state_argmax_before': x.argmax(-1).cpu(),
                              'endpoint_argmax': q.argmax(-1).cpu(),
                              'velocity_norm': ((q-x)/(1-s)).norm(dim=-1).cpu()})
            # Algebraically Euler for v=(q-x)/(1-s). At the last step gamma=1;
            # assigning q avoids cancellation and needs no state clipping.
            x = self._euler_update(x, q, s, t)
            if trace is not None:
                trace[-1]['state_argmax_after'] = x.argmax(-1).cpu()
        ids = torch.multinomial(x[0], 1).view(1, -1) if sample else x.argmax(-1)
        return ids, x

    @contextmanager
    def _seeded_rng(self, seed):
        device = self._generation_device()
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device], device_type=device.type):
            torch.manual_seed(seed)
            yield

    @contextmanager
    def _inference_eval(self):
        training = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(training)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Euler proposals followed by the existing exact AR verification rule."""
        kwargs.setdefault("jumps", self.cfg.train.get("val_decode_jumps", 4))
        with self._inference_eval():
            return super().generate(*args, **kwargs)

    @torch.no_grad()
    def sample_block(self, input_ids, *, block_size=32, steps=4, seed=0, return_trace=False):
        """Raw flow sample, without verification; prefix includes its last anchor.

        Use varying seeds to examine the drafter's joint distribution. Greedy
        projection is applied only after the last Euler step.
        """
        if self.cfg.get("decode", {}).get("fixed_prior", False):
            raise ValueError("sample_block requires decode.fixed_prior=false so seed controls the prior")
        with self._inference_eval(), self._seeded_rng(seed):
            ids = self._encode(None, input_ids)
            if ids.size(1) == 0:
                raise ValueError("sample_block needs a nonempty prefix including the anchor")
            cache = DynamicCache(config=self.orthrus.model.config)
            if ids.size(1) > 1:
                self.orthrus(ids[:, :-1], torch.ones_like(ids[:, :-1]), past_key_values=cache,
                             logits_to_keep=1)
            trace = [] if return_trace else None
            tokens, probabilities = self._draft_block(
                cache, block_size, self._jump_schedule(steps), anchor_token=ids[:, -1], trace=trace
            )
            result = {"tokens": tokens, "probabilities": probabilities}
            if trace is not None:
                result['trace'] = trace
            return result
