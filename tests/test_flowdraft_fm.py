"""Offline correctness checks with a randomly initialised, tiny Qwen3."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightning as L
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import Qwen3Config, Qwen3ForCausalLM, DynamicCache

from src.models.base.df_adapter import FlowDraftAttentionAdapter
from src.models.factory import build_lit
from src.models.flowdraft_fm import FlowDraftFM
from src.preprocessor.df_processor import DiffusionProcessor


def config():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "src/configs"),
                               version_base="1.3"):
        return compose("train", overrides=["+experiment=qwen06_flowdraft_fm"])


def mechanism():
    backbone = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=19, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, attention_dropout=0.0,
    ))
    backbone.config._attn_implementation = "eager"
    backbone.requires_grad_(False)
    adapter = FlowDraftAttentionAdapter(backbone, w_names=["q_proj", "k_proj", "v_proj"])
    return adapter, None, DiffusionProcessor(None, 19)


def model(anchors=2):
    cfg = config()
    cfg.trainer.max_steps = 2
    cfg.train.block_size = 4
    cfg.train.anchors_per_sequence = anchors
    cfg.train.val_decode_prompts = 0
    cfg.train.lr = 1e-3
    adapter, tokenizer, processor = mechanism()
    return FlowDraftFM(cfg, orthrus=adapter, tokenizer=tokenizer, df_processor=processor)


def batch():
    ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        [2, 5, 8, 4, 7, 3, 1, 6, 10, 12, 14]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FlowDraftFMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_config_requires_explicit_training_horizon(self):
        cfg = config()
        self.assertIsNone(cfg.trainer.max_steps)
        with self.assertRaisesRegex(ValueError, "trainer.max_steps"):
            FlowDraftFM(cfg)
        self.assertEqual(cfg.train.variant, "flowdraft_fm")
        self.assertEqual(list(cfg.model.adapter.w_names), ["q_proj", "k_proj", "v_proj"])
        self.assertEqual(list(cfg.train.acceptance_profile), [0.8])
        self.assertEqual(cfg.train.teacher_chain_tail_weight, 1.0)
        self.assertEqual(cfg.train.selfcorrect_tail_weight, 0.5)
        self.assertEqual(list(cfg.train.fm_train_steps), [1, 2, 3, 4])
        self.assertEqual(cfg.train.teacher_source, "corpus")
        self.assertEqual(cfg.train.verify_kl_weight, 0.0)
        self.assertEqual(cfg.train.selfcorrect_kl_weight, 1.0)

    def test_loss_and_gradients_equal_weighted_velocity_regression(self):
        logits = torch.randn(2, 3, 19, requires_grad=True)
        ids = torch.randint(19, (2, 3))
        live = torch.tensor([[True, True, False], [True, False, True]])
        x1, x0 = F.one_hot(ids, 19).float(), torch.randn_like(logits)
        t = torch.tensor([[0.0, 0.2, 0.4], [0.8, 0.9, 0.99]])
        x = (1 - t[..., None]) * x0 + t[..., None] * x1
        velocity = FlowDraftFM.velocity(logits.softmax(-1), x, t)
        reference = (((velocity - (x1 - x0)) * (1 - t[..., None])).square().sum(-1) * live).sum() / live.sum()
        actual = FlowDraftFM.endpoint_squared_loss(logits, ids, live)
        torch.testing.assert_close(actual, reference)
        g1, = torch.autograd.grad(actual, logits, retain_graph=True)
        g2, = torch.autograd.grad(reference, logits)
        torch.testing.assert_close(g1, g2)
        self.assertEqual(float(g1[~live].abs().sum()), 0.0)

    def test_multistep_kl_is_on_endpoint_of_same_velocity_field(self):
        m = model(1)
        logits = torch.randn(2, 37, 19, requires_grad=True)
        teacher = torch.randn_like(logits, requires_grad=True)
        live = torch.rand(2, 37) > 0.3
        weights = torch.rand(1, 37)
        x = torch.randn_like(logits)
        t = torch.linspace(0, 0.9, 37)[None].expand(2, -1)
        velocity = FlowDraftFM.velocity(logits.softmax(-1), x, t)
        endpoint = x + (1 - t[..., None]) * velocity
        p = teacher.detach().softmax(-1)
        per_token = (p * (p.log() - endpoint.log())).sum(-1)
        w = live * weights
        reference = (per_token * w).sum() / w.sum()
        actual = m._teacher_loss(teacher.detach(), logits, live, position_weight=weights)
        torch.testing.assert_close(actual, reference)
        g1, = torch.autograd.grad(actual, logits, retain_graph=True)
        g2, = torch.autograd.grad(reference, logits)
        torch.testing.assert_close(g1, g2, atol=1e-7, rtol=2e-5)
        self.assertEqual(float(g1[~live].abs().sum()), 0.0)
        self.assertEqual(torch.autograd.grad(actual, teacher, allow_unused=True), (None,))

    def test_packed_teacher_matches_individual_drafts_and_sequential_ar(self):
        m, b = model(), batch()
        anchors = torch.tensor([2, 6])
        with patch.object(m, "_sample_anchor_points", return_value=anchors), \
                patch.object(m.orthrus, "forward", wraps=m.orthrus.forward) as forward:
            corpus, targets, mask, live, cache, anchor, anchor_ids, geometry = m._prepare_fm(b)
        self.assertEqual(forward.call_count, 1)
        expected_ids = b["input_ids"][:, anchors[:, None] + torch.arange(1, 4)]
        torch.testing.assert_close(targets, expected_ids.flatten(1, 2))
        original = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        predictions = [torch.randn_like(corpus, requires_grad=True) for _ in range(4)]
        with patch.object(m.orthrus, "forward", wraps=m.orthrus.forward) as forward:
            packed = m._trajectory_teacher_logits(predictions, anchor_ids, mask, cache, geometry)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(forward.call_args.kwargs["weights"], "ar")
        with torch.no_grad(), m._teacher_eval():
            for logits, actual in zip(predictions, packed):
                draft = logits.argmax(-1)
                individual = m._verifier_response(draft, anchor_ids, mask, cache, geometry)
                torch.testing.assert_close(actual, individual, atol=1e-6, rtol=1e-5)
                for a, pos in enumerate(anchors):
                    ids = torch.cat([b["input_ids"][:, :pos + 1], draft.view(2, 2, 3)[:, a]], dim=-1)
                    expected = m.orthrus(ids, torch.ones_like(ids)).logits[:, pos:pos + 3]
                    torch.testing.assert_close(actual.view(2, 2, 3, -1)[:, a], expected,
                                               atol=1e-6, rtol=1e-5)
                self.assertFalse(actual.requires_grad)
        for layer, (keys, values) in zip(cache.layers, original):
            torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, values, rtol=0, atol=0)

    def test_training_updates_only_drafter_for_one_and_multiple_anchors(self):
        for count in (1, 2):
            with self.subTest(anchors=count):
                m, b = model(count), batch()
                before = {n: p.clone() for n, p in m.orthrus.model.named_parameters()}
                df_before = [p.clone() for p in m.orthrus.df_parameters()]
                optimizer = torch.optim.AdamW(m.orthrus.df_parameters(), lr=1e-3)
                loss, _ = m._fm_step(b)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is not None for p in m.orthrus.df_parameters()))
                self.assertTrue(all(p.grad is None for p in m.orthrus.model.parameters()))
                optimizer.step()
                self.assertTrue(any(not torch.equal(p, old) for p, old in zip(m.orthrus.df_parameters(), df_before)))
                for name, p in m.orthrus.model.named_parameters():
                    torch.testing.assert_close(p, before[name], rtol=0, atol=0)

    def test_multistep_preserves_original_kl_weights_and_previous_draft_targets(self):
        m = model(1).eval()

        def peaked(ids):
            return (4 * F.one_hot(torch.tensor([ids]), 19).float()).requires_grad_()

        predictions = [peaked([1, 2, 3]), peaked([2, 8, 5]), peaked([4, 5, 6])]
        teachers = [peaked([1, 8, 6]), peaked([7, 8, 5]), peaked([4, 5, 6])]
        live = torch.ones(1, 3, dtype=torch.bool)
        multi, accepted = m._multistep_term(live, predictions, teachers, 1)
        u = torch.tensor([[2.44, 1.44, 0.64]])

        def weighted_kl(p, z, w):
            p = p.detach().softmax(-1)
            per_token = (p * (p.log() - z.log_softmax(-1))).sum(-1)
            return (per_token * w).sum() / w.sum()

        expected = (
            weighted_kl(teachers[0], predictions[1], u * torch.tensor([[0.5, 1.0, 0.5]]))
            + weighted_kl(teachers[1], predictions[2], u * torch.tensor([[1.0, 0.5, 0.5]]))
        ) / 2
        torch.testing.assert_close(multi, expected)
        self.assertEqual(float(accepted), 3)
        actual_grad = torch.autograd.grad(multi, predictions[1:], retain_graph=True)
        expected_grad = torch.autograd.grad(expected, predictions[1:])
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(torch.autograd.grad(multi, [predictions[0], *teachers], allow_unused=True),
                         (None, None, None, None))

    def test_first_pass_corpus_gate_is_not_used(self):
        m = model(1)
        with patch.object(m, "_position_weights", side_effect=AssertionError("unexpected first-pass term")):
            loss, terms = m._fm_step(batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(terms), {"fm", "selfcorrect_kl", "accepted", "euler_steps"})

    def test_two_ar_calls_independent_of_block_length_and_loss_coefficients(self):
        for width, steps in ((4, 1), (6, 4)):
            with self.subTest(width=width, steps=steps):
                m = model(2)
                m.cfg.train.block_size = width
                m.cfg.train.fm_weight = 2.0
                m.cfg.train.selfcorrect_kl_weight = 3.0
                with patch.object(m, "_sample_train_steps", return_value=steps), \
                        patch.object(m.orthrus, "forward", wraps=m.orthrus.forward) as forward:
                    loss, terms = m._fm_step(batch())
                ar = [call for call in forward.call_args_list
                      if not call.kwargs.get("use_df", False) or call.kwargs.get("weights") == "ar"]
                self.assertEqual(len(ar), 2)
                self.assertEqual(forward.call_count, 2 + 1 + steps)
                torch.testing.assert_close(loss, 2 * terms["fm"] + 3 * terms["selfcorrect_kl"])
                self.assertEqual(float(terms["euler_steps"]), steps)
                if steps == 1:
                    self.assertEqual(float(terms["selfcorrect_kl"].detach()), 0)

    def test_training_trajectory_continues_euler_but_detaches_between_passes(self):
        m = model(1)
        x0 = torch.randn(1, 3, 19)
        snapshots = []
        outputs = []

        def field(x, t, *args):
            self.assertFalse(x.requires_grad)
            snapshots.append(x.clone())
            z = torch.randn_like(x, requires_grad=True)
            outputs.append(z)
            return z

        with patch.object(m, "_field_logits", side_effect=field):
            predictions = m._trajectory_logits(x0, 3, None, None, None, {})
        schedule = m._jump_schedule(3)
        x = x0
        for i, (s, t) in enumerate(schedule):
            torch.testing.assert_close(snapshots[i], x)
            x = x + (t - s) * FlowDraftFM.velocity(outputs[i].detach().softmax(-1), x, s)
        predictions[-1].sum().backward()
        self.assertIsNone(predictions[0].grad)
        self.assertIsNone(predictions[1].grad)
        self.assertIsNotNone(predictions[-1].grad)

    def test_empty_batch_still_checks_finiteness_and_connects_all_parameters(self):
        m = model()
        b = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
        with patch.object(m, "_assert_finite", wraps=m._assert_finite) as finite, patch.object(m, "log"):
            loss = m.training_step(b, 0)
        finite.assert_called_once()
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in m.orthrus.df_parameters()))

    def test_euler_follows_known_field_without_restarts(self):
        m = model(1)
        cache = DynamicCache(config=m.orthrus.model.config)
        x0 = torch.randn(1, 3, 19)
        logits = torch.randn_like(x0)
        endpoint = logits.softmax(-1)
        seen = []

        def field(x, t, anchor, mask, cache, geometry):
            seen.append(float(t))
            torch.testing.assert_close(x, (1 - t) * x0 + t * endpoint)
            self.assertEqual(int(anchor.argmax(-1)), 7)
            return logits

        with patch.object(m, "sample_prior", return_value=x0.clone()) as prior, patch.object(m, "_field_logits", side_effect=field):
            ids, q = m._draft_block(cache, 4, m._jump_schedule(4), anchor_token=torch.tensor([7]))
        prior.assert_called_once()
        self.assertEqual(seen, [0.0, 0.25, 0.5, 0.75])
        torch.testing.assert_close(q, endpoint)
        torch.testing.assert_close(ids, endpoint.argmax(-1))
        for schedule in ([[0, 1], [0.5, 1]], 0, -1):
            with self.assertRaises(ValueError):
                m._jump_schedule(schedule)

    def test_verified_generation_matches_ar_greedy_and_coupled(self):
        m = model()
        prompt = torch.tensor([[1, 4, 7, 9]])
        for steps in (1, 2, 4):
            for temperature in (0.0, 0.8):
                with self.subTest(steps=steps, temperature=temperature):
                    opts = dict(input_ids=prompt, max_new_tokens=11, temperature=temperature,
                                sampling_seed=73, top_k=8, top_p=0.9)
                    ar = m.ar_generate(**opts)
                    out = m.generate(**opts, block_size=4, jumps=steps)
                    self.assertEqual(out["new_tokens"], ar["new_tokens"])
        self.assertTrue(m.training)
        state = torch.random.get_rng_state().clone()
        a = m.sample_block(prompt, block_size=4, steps=4, seed=123)
        b = m.sample_block(prompt, block_size=4, steps=4, seed=123)
        torch.testing.assert_close(a["probabilities"], b["probabilities"], rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)

    def test_sampler_preserves_two_joint_modes_with_an_oracle_field(self):
        m = model(1)
        cache = DynamicCache(config=m.orthrus.model.config)
        modes = torch.tensor([[2, 3], [5, 7]])
        endpoints = F.one_hot(modes, 19).float()

        def oracle(x, t, *args):
            distance = (x[:, None] - t * endpoints[None]).square().sum((-1, -2))
            posterior = (-distance / (2 * (1 - t)**2)).softmax(-1)
            q = torch.einsum("bm,mlv->blv", posterior, endpoints)
            return q.clamp_min(1e-30).log()

        counts = [0, 0]
        with patch.object(m, "_field_logits", side_effect=oracle):
            for seed in range(32):
                with m._seeded_rng(seed):
                    ids, _ = m._draft_block(cache, 3, m._jump_schedule(32), anchor_token=torch.tensor([1]))
                match = (modes == ids).all(-1)
                self.assertTrue(bool(match.any()), f"invalid mixed block: {ids}")
                counts[int(match.long().argmax())] += 1
        self.assertTrue(all(4 <= count <= 28 for count in counts), counts)

    def test_lightning_training_and_factory_checkpoint_roundtrip(self):
        m = model(1)
        m.cfg.train.val_decode_prompts = 1
        m.cfg.train.val_decode_max_new = 5
        examples = [{key: value[i] for key, value in batch().items()} for i in range(2)]
        loader = DataLoader(examples, batch_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = L.Trainer(accelerator="cpu", devices=1, max_steps=2,
                                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                                enable_model_summary=False, limit_val_batches=1,
                                val_check_interval=1, num_sanity_val_steps=1,
                                default_root_dir=tmp)
            trainer.fit(m, loader, loader)
            self.assertTrue(torch.isfinite(trainer.callback_metrics["val/loss"]))
            self.assertTrue(torch.isfinite(trainer.callback_metrics["val/tpf"]))
            path = Path(tmp) / "fm.ckpt"
            trainer.save_checkpoint(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(all("orthrus.model." not in name for name in saved["state_dict"]))
            cfg = OmegaConf.create(OmegaConf.to_container(m.cfg))
            cfg.checkpoint = str(path)
            with patch("src.models.flowdraft.build_model", return_value=mechanism()):
                restored = build_lit(cfg)
            self.assertIsInstance(restored, FlowDraftFM)
            for name, parameter in restored.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter, saved["state_dict"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
