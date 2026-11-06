"""CPU checks for the Qwen fixed-bank experiment and its restart contract."""

import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import lightning as L
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.data.fm_bank import EndpointBank, FixedTokenPool
from src.models.flowdraft_fm_bank import FlowDraftFMBank
from test_flowdraft_fm import batch, mechanism


def config():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'src/configs'), version_base='1.3'):
        return compose('train', overrides=['+experiment=qwen06_flowdraft_fm_bank'])


class BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def model(self):
        cfg = config()
        cfg.train.block_size = 4
        cfg.train.fm_bank_size = 4
        cfg.train.fm_bank_path = str(Path(self.tmp.name) / 'bank.sqlite')
        adapter, tokenizer, processor = mechanism()
        m = FlowDraftFMBank(cfg, orthrus=adapter, tokenizer=tokenizer, df_processor=processor)
        self.addCleanup(lambda: m._bank_store.close() if m._bank_store is not None else None)
        return m

    def test_authorized_budget_and_geometry(self):
        cfg = config()
        self.assertEqual(cfg.trainer.max_steps, 60000)
        self.assertEqual(cfg.data.train_size, 80000)
        self.assertEqual(cfg.train.anchors_per_sequence, 4)
        self.assertEqual(cfg.train.teacher_source, 'fixed_ar_bank')
        self.assertEqual(cfg.data.batch_size * cfg.trainer.accumulate_grad_batches, 16)

    def test_bank_persistence_and_incompatible_specification_rejected(self):
        path = Path(self.tmp.name) / 'data.sqlite'
        spec = {'size': 3, 'length': 2}
        store = EndpointBank(path, spec)
        prefix = torch.tensor([1, 3, 2])
        ids = torch.tensor([[1, 1], [1, 1], [0, 0]])
        store.put(prefix, ids)
        store.close()
        store = EndpointBank(path, spec)
        torch.testing.assert_close(store.get(prefix), ids)
        self.assertIsNone(store.get(torch.tensor([1, 2, 3])))
        store.close()
        with self.assertRaisesRegex(ValueError, 'specification'):
            EndpointBank(path, {**spec, 'size': 4})

    def test_sparse_posterior_and_kl_match_dense_value_and_gradients(self):
        bank = torch.tensor([[[[0, 0], [0, 0], [1, 2]], [[2, 1], [0, 2], [1, 2]]]])
        x = torch.randn(1, 4, 3)
        post = FlowDraftFMBank.bank_posterior_weights(x, 0.7, bank, noise_std=1.3)
        endpoints = F.one_hot(bank, 3).float()
        loglike = -(x.view(1, 2, 1, 2, 3) - 0.7 * endpoints).square().sum((-1, -2)) / (2 * 1.3**2 * 0.3**2)
        torch.testing.assert_close(post, loglike.softmax(-1))
        target = (post[..., None, None] * endpoints).sum(2).flatten(1, 2)
        logits = torch.randn_like(x, requires_grad=True)
        live = torch.tensor([[True, True, False, True]])
        w = torch.tensor([[0.7, 2.0, 0.4, 1.3]])
        expected = (torch.xlogy(target, target) - target * logits.log_softmax(-1)).sum(-1)
        expected = (expected * live * w).sum() / (live * w).sum()
        actual = FlowDraftFMBank.bank_kl(logits, bank, post, live, w)
        torch.testing.assert_close(actual, expected)
        g1, = torch.autograd.grad(actual, logits, retain_graph=True)
        g2, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(g1, g2)
        self.assertFalse(post.requires_grad)

    def test_bank_draws_match_sequential_ar_with_same_randomness(self):
        m = self.model()
        prefix = torch.tensor([1, 3, 5])
        bank = m._sample_ar_bank(prefix)
        digest, _ = EndpointBank.identity(prefix)
        rng = torch.Generator().manual_seed((int(digest[:15], 16) + int(m.cfg.seed)) % (2**63-1))
        seq = prefix[None].expand(4, -1).clone()
        with torch.no_grad(), m._teacher_eval():
            for _ in range(3):
                logits = m.orthrus(seq, torch.ones_like(seq)).logits[:, -1]
                y = torch.multinomial(logits.float().softmax(-1), 1, generator=rng)
                seq = torch.cat([seq, y], -1)
        torch.testing.assert_close(bank, seq[:, -3:])

    def test_four_anchor_backward_preserves_teacher_and_reuses_banks(self):
        m = self.model()
        before = {n: p.clone() for n, p in m.orthrus.model.named_parameters()}
        with patch.object(m, '_sample_train_steps', return_value=4):
            loss, terms = m._fm_step(batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(terms['bank_new_prefixes']), 8)
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in m.orthrus.df_parameters()))
        self.assertTrue(all(p.grad is None for p in m.orthrus.model.parameters()))
        for n, p in m.orthrus.model.named_parameters():
            torch.testing.assert_close(before[n], p, atol=0, rtol=0)
        with patch.object(m, '_sample_ar_bank', side_effect=AssertionError('bank regenerated')):
            prepared = m._prepare_bank(batch())
        self.assertEqual(prepared[-2], 0)

    def test_four_anchor_verifier_respects_individual_prefixes(self):
        m, b = self.model(), batch()
        bank, anchor, anchor_ids, mask, cache, geometry, *_ = m._prepare_bank(b)
        draft = bank[:, :, 0].flatten(1)
        teacher = m._verifier_response(draft, anchor_ids, mask, cache, geometry).view(2, 4, 3, 19)
        positions = geometry['position_ids'].view(2, 4, 4)[:, :, 0]
        with torch.no_grad(), m._teacher_eval():
            for row, a in itertools.product(range(2), range(4)):
                p = int(positions[row, a])
                seq = torch.cat([b['input_ids'][row, :p+1], bank[row, a, 0]])[None]
                expected = m.orthrus(seq, torch.ones_like(seq)).logits[0, p:p+3]
                torch.testing.assert_close(teacher[row, a], expected, atol=1e-6, rtol=1e-5)

    def test_infinite_pool_resume_preserves_membership_and_order(self):
        rows = {'input_ids': torch.arange(7)[:, None], 'attention_mask': torch.ones(7, 1)}
        original = list(itertools.islice(iter(FixedTokenPool(rows, 17)), 31))
        resumed = list(itertools.islice(iter(FixedTokenPool(rows, 17, skip=19)), 12))
        self.assertEqual([int(x['input_ids']) for x in original[19:]], [int(x['input_ids']) for x in resumed])
        for start in (0, 7, 14):
            self.assertEqual(sorted(int(x['input_ids']) for x in original[start:start+7]), list(range(7)))

    def test_sampler_trace_is_observational_and_inference_needs_no_bank(self):
        m = self.model()
        prefix = torch.tensor([[1, 3, 5]])
        with patch.object(m, '_store', side_effect=AssertionError('inference used bank')):
            plain = m.sample_block(prefix, block_size=4, steps=4, seed=123)
            traced = m.sample_block(prefix, block_size=4, steps=4, seed=123, return_trace=True)
        torch.testing.assert_close(plain['probabilities'], traced['probabilities'], atol=0, rtol=0)
        self.assertEqual(len(traced['trace']), 4)
        self.assertEqual([(v['t_start'],v['t_end']) for v in traced['trace']], [(0,.25),(.25,.5),(.5,.75),(.75,1)])
        torch.testing.assert_close(traced['tokens'].cpu(), traced['trace'][-1]['state_argmax_after'])

    def test_lightning_resume_keeps_bank_optimizer_and_60k_schedule(self):
        class StopAt(L.Callback):
            def __init__(self, step):
                self.step = step
            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                if trainer.global_step >= self.step:
                    trainer.should_stop = True
        rows = batch()
        loader = torch.utils.data.DataLoader(FixedTokenPool(rows, seed=42), batch_size=1)
        def trainer(stop):
            return L.Trainer(max_steps=60000, max_epochs=-1, accelerator='cpu', devices=1,
                             logger=False, enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, limit_val_batches=0,
                             callbacks=[StopAt(stop)])
        torch.manual_seed(16)
        first = self.model()
        t1 = trainer(2)
        t1.fit(first, loader)
        path = str(Path(self.tmp.name) / 'resume.ckpt')
        t1.save_checkpoint(path, weights_only=False)
        checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint['fm_bank']['prefixes'], 8)
        self.assertEqual(checkpoint['lr_schedulers'][0]['last_epoch'], 2)
        torch.manual_seed(16)
        second = self.model()
        t2 = trainer(3)
        t2.fit(second, loader, ckpt_path=path)
        self.assertEqual(t2.global_step, 3)
        self.assertEqual(t2.lr_scheduler_configs[0].scheduler.last_epoch, 3)
        expected_lr = float(second.cfg.train.lr) * 3 / 3000
        self.assertAlmostEqual(t2.optimizers[0].param_groups[0]['lr'], expected_lr)
        second.cfg.trainer.max_steps = 80000
        with self.assertRaisesRegex(ValueError, 'horizon'):
            second.on_load_checkpoint(checkpoint)


if __name__ == '__main__':
    unittest.main()
