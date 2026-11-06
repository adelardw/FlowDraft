"""FM and weighted refinement targeting one fixed empirical AR law per prefix."""

import json
import time

import torch
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from src.data.fm_bank import EndpointBank
from src.models.flowdraft_fm import FlowDraftFM


class FlowDraftFMBank(FlowDraftFM):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        if cfg.train.teacher_source != 'fixed_ar_bank':
            raise ValueError('flowdraft_fm_bank requires teacher_source=fixed_ar_bank')
        if int(cfg.train.fm_bank_size) < 2:
            raise ValueError('fm_bank_size must be at least two')
        self._bank_store = None

    def _store(self):
        if self._bank_store is None:
            backbone = self.orthrus.model.config
            specification = {
                'version': 1, 'model': str(self.cfg.model.name),
                'revision': str(getattr(backbone, '_commit_hash', None)),
                'config': backbone.to_dict(),
                'dtype': str(next(self.orthrus.model.parameters()).dtype),
                'size': int(self.cfg.train.fm_bank_size),
                'length': int(self.cfg.train.block_size) - 1,
                'seed': int(self.cfg.seed), 'temperature': 1.0,
                'top_k': None, 'top_p': None, 'stop_at_eos': False,
            }
            self._bank_store = EndpointBank(self.cfg.train.fm_bank_path, specification)
        return self._bank_store

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        store = self._store()
        checkpoint['fm_bank'] = {
            'prefixes': store.db.execute('SELECT COUNT(*) FROM banks').fetchone()[0],
            'specification': store.specification,
        }

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint['hyper_parameters']
        if saved['trainer']['max_steps'] != self.cfg.trainer.max_steps:
            raise ValueError('FM resume cannot change the original LR horizon')
        for key in ('anchors_per_sequence', 'fm_bank_size', 'block_size'):
            if saved['train'][key] != self.cfg.train[key]:
                raise ValueError(f'FM resume cannot change {key}')
        metadata = checkpoint.get('fm_bank')
        if metadata is None:
            raise ValueError('checkpoint is missing its fixed-bank metadata')
        store = self._store()
        if json.dumps(metadata['specification'], sort_keys=True) != json.dumps(store.specification, sort_keys=True):
            raise ValueError('checkpoint and endpoint bank have different specifications')
        if store.db.execute('SELECT COUNT(*) FROM banks').fetchone()[0] < metadata['prefixes']:
            raise ValueError('endpoint bank is missing entries required by this checkpoint')
        super().on_load_checkpoint(checkpoint)

    @torch.no_grad()
    def _sample_ar_bank(self, prefix):
        count = int(self.cfg.train.fm_bank_size)
        length = int(self.cfg.train.block_size) - 1
        digest, _ = EndpointBank.identity(prefix)
        seed = (int(digest[:15], 16) + int(self.cfg.seed)) % (2**63 - 1)
        rng = torch.Generator(device=prefix.device).manual_seed(seed)
        cache = DynamicCache(config=self.orthrus.model.config)
        with self._teacher_eval():
            logits = self.orthrus(prefix[None], torch.ones_like(prefix[None]),
                                  past_key_values=cache, logits_to_keep=1).logits[:, -1]
            cache.batch_repeat_interleave(count)
            logits = logits.expand(count, -1)
            tokens = []
            for j in range(length):
                token = torch.multinomial(logits.float().softmax(-1), 1, generator=rng)
                tokens.append(token[:, 0])
                if j + 1 < length:
                    mask = torch.ones(count, prefix.numel() + j + 1, dtype=torch.long, device=prefix.device)
                    logits = self.orthrus(token, mask, past_key_values=cache,
                                          logits_to_keep=1).logits[:, -1]
        return torch.stack(tokens, -1)

    def _prepare_bank(self, batch):
        ids, mask = batch['input_ids'], batch['attention_mask']
        width, count = int(self.cfg.train.block_size), int(self.cfg.train.anchors_per_sequence)
        positions, banks = [], []
        generated, started = 0, time.perf_counter()
        store = self._store()
        for row, row_mask in zip(ids, mask):
            length = int(row_mask.sum())
            if length < count + 2 or not bool(row_mask[:length].all()) or bool(row_mask[length:].any()):
                raise ValueError('fixed-bank rows require a right-padded prefix with enough distinct anchors')
            digest, _ = EndpointBank.identity(row[:length])
            rng = torch.Generator().manual_seed(int(digest[:15], 16) + int(self.cfg.seed))
            # Keep the first token as context. Each selected anchor is clean.
            anchors = (torch.randperm(length - 1, generator=rng)[:count] + 1).sort().values
            positions.append(anchors)
            row_banks = []
            for anchor in anchors.tolist():
                prefix = row[:anchor + 1]
                bank = store.get(prefix)
                if bank is None:
                    bank = self._sample_ar_bank(prefix).cpu()
                    store.put(prefix, bank)
                    generated += 1
                row_banks.append(bank)
            banks.append(torch.stack(row_banks))
        anchors = torch.stack(positions).to(ids.device)
        banks = torch.stack(banks).to(ids.device)
        cache = DynamicCache(config=self.orthrus.model.config)
        with torch.no_grad(), self._teacher_eval():
            self.orthrus(ids, mask, past_key_values=cache, logits_to_keep=1)
        anchor_ids = ids.gather(1, anchors)
        anchor = self.df_processor.to_simplex(anchor_ids)
        geometry = {
            'position_ids': (anchors[..., None] + torch.arange(width, device=ids.device)).flatten(1),
            'causal_limit': (anchors - 1).repeat_interleave(width, dim=1),
            'diffusion_block_size': width,
        }
        return banks, anchor, anchor_ids, mask, cache, geometry, generated, time.perf_counter() - started

    @staticmethod
    @torch.no_grad()
    def bank_posterior_weights(x, t, bank, noise_std=1.0):
        if not 0 <= t < 1:
            raise ValueError('posterior time must be in [0,1)')
        batch, anchors, _, length = bank.shape
        state = x.reshape(batch, anchors, length, -1).float()
        selected = state.gather(-1, bank.transpose(-1, -2)).sum(-2)
        return (selected * (t / (noise_std**2 * (1-t)**2))).softmax(-1)

    @staticmethod
    def bank_kl(logits, bank, posterior, live, position_weights):
        """Exact empirical marginal KL, including duplicate endpoint token IDs."""
        batch, anchors, size, length = bank.shape
        ids = bank.transpose(-1, -2).reshape(-1, size)
        probabilities = posterior.detach()[:, :, None, :].expand(-1, -1, length, -1).reshape(-1, size)
        weights = (live.float() * position_weights.detach()).flatten()
        flat = logits.reshape(-1, logits.size(-1))

        def term(z, tokens, p, w):
            log_q = z.float().log_softmax(-1).gather(-1, tokens)
            same = tokens[:, :, None] == tokens[:, None, :]
            target_at_token = (same * p[:, None, :]).sum(-1)
            error = (p * (target_at_token.clamp_min(1e-30).log() - log_q)).sum(-1)
            return (error * w).sum()

        total = flat.new_zeros((), dtype=torch.float32)
        for start in range(0, flat.size(0), 32):
            args = (flat[start:start+32], ids[start:start+32], probabilities[start:start+32], weights[start:start+32])
            total = total + (checkpoint(term, *args, use_reentrant=False) if torch.is_grad_enabled() else term(*args))
        return total / weights.sum().clamp_min(1)

    def _fm_step(self, batch):
        bank, anchor, anchor_ids, mask, cache, geometry, misses, seconds = self._prepare_bank(batch)
        b, a, size, length = bank.shape
        choice = torch.randint(size, (b, a, 1, 1), device=bank.device).expand(-1, -1, 1, length)
        targets = bank.gather(2, choice).squeeze(2).flatten(1)
        live = torch.ones_like(targets, dtype=torch.bool)
        x1 = self.df_processor.to_simplex(targets)
        x0 = self.sample_prior(x1)
        times = torch.rand(b, a, device=bank.device).repeat_interleave(length, dim=1)
        x = (1-times[..., None]) * x0 + times[..., None] * x1
        logits = self._field_logits(x, times, anchor, mask, cache, geometry)
        fm = self.endpoint_squared_loss(logits, targets, live)
        del x, x0, logits

        steps = self._sample_train_steps(bank.device)
        state = self.sample_prior(x1)
        predictions, posteriors = [], []
        for s, t in self._jump_schedule(steps):
            posteriors.append(self.bank_posterior_weights(state, s, bank, self.noise_std))
            z = self._field_logits(state, s, anchor, mask, cache, geometry)
            predictions.append(z)
            state = self._euler_update(state, z.detach().float().softmax(-1), s, t).detach()
        teachers = self._trajectory_teacher_logits(predictions, anchor_ids, mask, cache, geometry)
        terms = []
        for i, (prediction, teacher) in enumerate(zip(predictions, teachers)):
            accepted, known = self._greedy_verdict(prediction.detach().argmax(-1), teacher.argmax(-1), live, a)
            if i + 1 < steps:
                weights = self._refinement_weights(accepted, known, live, a)
                terms.append(self.bank_kl(predictions[i+1], bank, posteriors[i+1], live, weights))
        ms = torch.stack(terms).mean() if terms else predictions[0].sum() * 0
        accepted_mean = accepted.reshape(b, a, length).sum(-1).float().mean()
        loss = float(self.cfg.train.fm_weight) * fm + float(self.cfg.train.selfcorrect_kl_weight) * ms
        return loss, {'fm': fm, 'selfcorrect_kl': ms, 'accepted': accepted_mean,
                      'euler_steps': fm.new_tensor(steps), 'bank_new_prefixes': fm.new_tensor(misses),
                      'bank_prepare_seconds': fm.new_tensor(seconds)}
