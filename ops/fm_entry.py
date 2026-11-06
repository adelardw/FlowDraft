"""Bounded Kaggle training session; the optimizer horizon stays at 60k."""
import json
import os
import sys
import time
from pathlib import Path

import lightning as L
import torch
import src.train as entry


def serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


class Session(L.Callback):
    def on_train_start(self, trainer, module):
        self.root = Path(module.cfg.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.start = self.last_time = time.monotonic()
        self.before = self.last_step = int(trainer.global_step)
        self.finite_updates = 0
        self.overflows = 0
        self.overflow_streak = 0
        if self.before == 0 and trainer.precision_plugin.scaler is not None:
            scale_state = trainer.precision_plugin.scaler.state_dict()
            scale_state['scale'] = 128.0
            trainer.precision_plugin.scaler.load_state_dict(scale_state)
            print('FM_INITIAL_LOSS_SCALE', 128.0, flush=True)
        assert trainer.max_steps == 60000
        assert module.cfg.data.train_size == 80000
        assert module.cfg.train.anchors_per_sequence == 4
        rows = torch.load(module.cfg.data.fm_tokens_path, map_location='cpu', weights_only=True)
        batch = {k: v[:1].long().to(module.device) for k,v in rows['train'].items()}
        del rows
        choices = module.train_step_choices
        torch.cuda.reset_peak_memory_stats()
        with torch.random.fork_rng(devices=[module.device]):
            try:
                module.train_step_choices = (4,)
                with trainer.precision_plugin.forward_context():
                    loss, terms = module._fm_step(batch)
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in module.orthrus.df_parameters()):
                    raise RuntimeError('four-pass preflight produced nonfinite loss/gradients')
                module.zero_grad(set_to_none=True)
                prompt = batch['input_ids'][:, :32]
                with trainer.precision_plugin.forward_context():
                    sample = module.sample_block(prompt, steps=4, seed=17, return_trace=True)
                    trace = {k: v for k,v in sample.items() if k != 'probabilities'}
                    if not bool(torch.isfinite(sample['probabilities']).all()):
                        raise RuntimeError('sampler produced nonfinite probabilities')
                    for temperature in (0.0, 0.8):
                        opts = dict(input_ids=prompt, max_new_tokens=8, temperature=temperature, sampling_seed=73)
                        reference = module.ar_generate(**opts)
                        verified = module.generate(**opts, block_size=32, jumps=4)
                        if reference['new_tokens'] != verified['new_tokens']:
                            raise RuntimeError('verified sampling differs from the frozen AR')
                (self.root/'sampling-preflight.json').write_text(json.dumps(trace, default=serial, indent=2))
                print('FM_PREFLIGHT_SUCCESS', json.dumps({
                    'loss': float(loss.detach()), 'peak_gib': torch.cuda.max_memory_allocated()/2**30,
                    'four_anchors': True, 'four_passes': True, 'ar_greedy_and_coupled_match': True,
                    'max_steps': trainer.max_steps, 'step_before': self.before,
                }), flush=True)
                del loss, terms, sample
            finally:
                module.train_step_choices = choices
        torch.cuda.empty_cache()
        self.last_time = time.monotonic()

    def on_before_optimizer_step(self, trainer, module, optimizer):
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.orthrus.df_parameters())
        if finite:
            self.finite_updates += 1
            self.overflow_streak = 0
        else:
            self.overflows += 1
            self.overflow_streak += 1
            if self.overflow_streak > 8:
                raise RuntimeError('repeated FP16 gradient overflow')

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == self.last_step:
            return
        torch.cuda.synchronize()
        now = time.monotonic()
        row = {'step': step, 'target': trainer.max_steps, 'seconds': now-self.last_time,
               'finite_updates': self.finite_updates, 'overflows': self.overflows,
               'lr': trainer.optimizers[0].param_groups[0]['lr'],
               'loss_scale': trainer.precision_plugin.scaler.get_scale(),
               'peak_gib': torch.cuda.max_memory_allocated()/2**30}
        for key in ('loss/fm','loss/selfcorrect_kl','loss/accepted','loss/bank_prepare_seconds'):
            if key in trainer.callback_metrics:
                row[key] = float(trainer.callback_metrics[key])
        with (self.root/'progress.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if self.finite_updates <= 5 or step-self.before <= 5 or step % 10 == 0:
            print('FM_PROGRESS', json.dumps(row), flush=True)
        self.last_step, self.last_time = step, now
        if now-self.start >= float(os.environ.get('FD_FM_SESSION_HOURS','3'))*3600:
            trainer.should_stop = True

    def on_train_end(self, trainer, module):
        if self.finite_updates == 0:
            raise RuntimeError('no finite optimizer updates in this session')
        summary = {'step_before': self.before, 'step_after': int(trainer.global_step),
                   'target': trainer.max_steps, 'finite_updates': self.finite_updates,
                   'elapsed_seconds': time.monotonic()-self.start,
                   'bank_prefixes': module._store().db.execute('SELECT COUNT(*) FROM banks').fetchone()[0]}
        (self.root/'session-summary.json').write_text(json.dumps(summary, indent=2))
        print('FM_SESSION_COMPLETE', json.dumps(summary), flush=True)


original_callbacks = entry.build_checkpoint_callbacks
entry.build_checkpoint_callbacks = lambda cfg, **kw: [*original_callbacks(cfg, **kw), Session()]
exit_code = 0
try:
    entry.main()
except SystemExit as error:
    exit_code = int(error.code or 0)
except BaseException:
    import traceback
    traceback.print_exc()
    exit_code = 1
sys.stdout.flush()
sys.stderr.flush()
os._exit(exit_code)
