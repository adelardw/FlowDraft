# FlowDraft: Flow-Map Drafting for Lossless Parallel Decoding

> Raising the **acceptance ceiling** of lossless parallel decoding by upgrading the *drafter* to a **Categorical Flow Map** — faster generation, provably identical output.

**Documentation:** **English** · [Russian](README.ru.md)

<!-- Badges — TODO: fill in once the repo is public
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-WIP-orange)
-->

> 🚧 **Status: core implementation landed and smoke-verified** — adapter, four training variants (`flowdraft`, `flowdraft_block_wise`, `orthrus`, and `orthrus_block_wise`; trained end-to-end on real data: SmolLM2-135M + Nemotron), lossless decoding (**bitwise** at greedy AND at sampling via Gumbel coupling; `jumps+1` forwards per cycle), evaluation harness (mean±std, JSONL results + report plots), experiment presets for every stage of the task. GPU experiments pending; **Results** are TBD.

**Summer of Machine Learning at Skoltech (SMILES) · Applied AI Center**

---


## Table of contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Experiments (task stages)](#experiments-task-stages)
- [Background: the decoding bottleneck](#background-the-decoding-bottleneck)
- [Host framework: Orthrus](#host-framework-orthrus)
- [The problem](#the-problem)
- [Key idea: a Categorical Flow Map drafter](#key-idea-a-categorical-flow-map-drafter)
- [CFM training, in brief](#cfm-training-in-brief)
- [Goals](#goals)
- [Expected deliverables](#expected-deliverables)
- [Method](#method) 🚧
- [Repository structure](#repository-structure)
- [Installation](#installation) 🚧
- [Usage](#usage) 🚧
- [Training](#training) 🚧
- [Configuration reference](#configuration-reference)
- [Inference parameters, in plain words](#inference-parameters-in-plain-words)
- [Evaluation](#evaluation) 🚧
- [Results](#results) 🚧
- [References](#references)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [License](#license) 🚧

## Overview

Autoregressive (AR) LLMs decode strictly sequentially: generating *L* tokens costs *L* forward passes, which is memory-bandwidth bound. Diffusion LMs can draft whole blocks in parallel, but they drift from the AR distribution and lose quality. Speculative-style verification restores quality: draft a block in parallel, then verify it against the AR model in a single pass and keep only the tokens the AR model would have produced — this is **lossless**.

**FlowDraft** upgrades the *drafter* inside a lossless parallel-decoding loop. The throughput of any verify-based system is governed by its **acceptance length** — the number of drafted tokens accepted per cycle. We replace the single-step masked-diffusion drafter with a **Categorical Flow Map** drafter that produces a higher-fidelity *joint* proposal over the block at the **same** number of forward passes. Verification is left untouched, so the output stays strictly lossless — the drafter affects only **speed**, never **quality**.

Crucially, the AR model is what does the verifying, so it is kept **frozen throughout**. Keeping it untouched is exactly what makes the output provably identical to the base model; it is what the word *lossless* rests on.

## Quickstart

```bash
# 1. Setup (once)
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env        # gated meta-llama access
./hf-auth.sh                         # verify: prints your HF username

# 2. Check inference works — the UNTRAINED drafter is already lossless (just slow)
./hf-auth.sh uv run python main.py -p "Once upon a time"
#    -> generation + [lossless vs greedy AR: PASS]

# 3. Train FlowDraft (full-sequence recipe; GPU recommended)
./hf-auth.sh uv run python src/train.py \
    trainer.max_steps=10000 data.batch_size=8
#    watch: loss/endpoint ↓, loss/ec ↓, loss/td sane, val/teacher_agreement ↑
#    checkpoints (DF head + Adam moments, ~5 GB for 3B) land in checkpoints/
#    our ADDITION beyond the task — training in the exact inference geometry:
#    append train.variant=flowdraft_block_wise
#    epochs on top of streaming (nothing is downloaded ahead): a fixed pool of
#    N samples repeated M times, each repetition in a new order —
#    ./hf-auth.sh uv run python src/train.py \
#        data.train_size=471952 trainer.max_epochs=2 trainer.max_steps=7375

For multi-GPU training, let Lightning run DDP and specify the GPU count:

```bash
./hf-auth.sh uv run python src/train.py \
    trainer.accelerator=gpu trainer.devices=2 trainer.strategy=ddp \
    data.batch_size=8 trainer.max_steps=10000
```

Training always disables `model.backbone.device_map`: Hugging Face device maps are inference sharding, while DDP needs one complete model replica per GPU. The shown `data.batch_size` is per GPU; use `trainer.accumulate_grad_batches` to reach a larger effective global batch.

## H100 sparse-attention setup

The paper-style `+experiment=orthrus` uses PyTorch **FlexAttention** for its
256 isolated masked blocks. On Hopper (H100/H200) and Blackwell, it selects
FlexAttention's `FLASH` backend (the FlashAttention-4 path); on older CUDA
GPUs it falls back to FlexAttention's Triton backend with the same sparse-mask
semantics but lower throughput.

Install FA4 manually on the CUDA node (it is deliberately not a project
dependency, so CPU/macOS development environments remain lightweight):

```bash
uv pip install ninja packaging
# FA4 is currently distributed as a prerelease.
uv pip install --prerelease=allow --no-build-isolation "flash-attn-4[cu13]"
```

Verify the required training API before launching a run:

```bash
uv run python -c "from torch.nn.attention.flex_attention import flex_attention; print('FlexAttention: OK')"
uv run python -c "import flash_attn; print('flash-attn: OK')"
```

`flash-attn-4` accelerates the Hopper/Blackwell backend; FlexAttention is the
required component for sparse baseline training. CPU, Apple Silicon, and
CUDA builds without FlexAttention use neither of these paths.

# 4. Measure acceptance / TPF vs the AR baseline (lossless asserted bitwise)
./hf-auth.sh uv run python src/eval.py checkpoint=checkpoints/last.ckpt

# 5. Generate with the trained drafter (greedy; add --temperature for sampling)
./hf-auth.sh uv run python main.py -p "..." \
    --checkpoint checkpoints/last.ckpt --jumps 2
```

Laptop debugging: `src/train.py` and `src/eval.py` also run on a small ungated
backbone — append the hydra overrides
`model.name=HuggingFaceTB/SmolLM2-135M-Instruct model.backbone.dtype=float32
model.backbone.device_map=null`.

## Experiments (task stages)

Every stage of the project task is one preset in `src/configs/experiment/`. The
**detailed walkthrough — what each stage means, which training curves to
watch, expected behaviour, results-table rows, and the analysis pipeline —
is the [Russian experiment walkthrough](README.ru.md).**
The command summary:

```bash
# Stage 1 — reproduce the Orthrus masked-diffusion baseline @ 0.5B
./hf-auth.sh uv run python src/train.py +experiment=orthrus

# Stages 2-3 — flow-map drafter, staged VFM/ECLD (endpoint first, consistency ramped in)
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_staged

# Stage 4 — lossless at sampling: coupled = bitwise; uncoupled = null-calibrated TV test
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b checkpoint=<ckpt> decode.temperature=0.8
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b checkpoint=<ckpt> decode.temperature=0.8 \
    decode.coupled=false decode.equiv_samples=500

# Stage 5 (ablations) — the contribution of each distillation term
./hf-auth.sh uv run python src/train.py +experiment=ablate_teacher_only
./hf-auth.sh uv run python src/train.py +experiment=ablate_consistency_only

# Stage 5 (evaluation) — default: MATH-500, a dataset never seen in training
#           (data=nemotron evaluates on the training distribution);
#           block-size x jumps grid -> results/eval.jsonl -> report figures.
#           NOTE: decode.block_size = K, the number of tokens drafted per cycle at
#           INFERENCE — a knob of EVERY variant; it is unrelated to the
#           flowdraft_block_wise TRAINING-geometry variant despite the similar name
./hf-auth.sh uv run python src/eval.py -m model=qwen3_1.7b variant=flowdraft checkpoint=<ckpt> \
    decode.block_size=4,8,16 decode.jumps=1,2,4
uv run python src/plots.py

# ADDITION (beyond the task) — both drafters retrained in the exact inference
# geometry (block-causal): the geometry-for-geometry comparison
./hf-auth.sh uv run python src/train.py +experiment=orthrus_block_wise
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_block_wise
#   eval: same stage-5 commands with variant=orthrus_block_wise / flowdraft_block_wise
```

Training curves land in TensorBoard (`uv run tensorboard --logdir checkpoints`);
metric formulas and the training↔eval correspondence are spelled out in the [Russian guide](README.ru.md).

To mirror the same metrics to Weights & Biases, authenticate once with
`uv run wandb login` (or set `WANDB_API_KEY`) and enable the opt-in sink:

```bash
./hf-auth.sh uv run python src/train.py \
    wandb.enabled=true wandb.project=flowdraft wandb.name=qwen2-flowdraft
```

Use `wandb.offline=true` to record locally and upload later with `wandb sync`.

## Background: the decoding bottleneck

- **AR LLMs** decode strictly sequentially: *L* tokens → *L* forward passes (memory-bandwidth bound).
- **Diffusion LMs** draft blocks in parallel, but drift from the AR distribution and lose quality.
- **Speculative-style verification** fixes quality: draft in parallel, then *verify* against the AR model → keep only correct tokens (**lossless**).

## Host framework: Orthrus

FlowDraft is built inside **Orthrus**, a lossless parallel-decoding scaffold:

- One transformer, two attention paths: a **frozen AR path** and a **lightweight, trainable diffusion path** (~16% of parameters), sharing the same norm / MLP / embeddings and a single KV cache.
- The diffusion path proposes *K* tokens in parallel; the frozen AR head verifies them in one pass → output **provably identical** to the base model. Accepted tokens are committed to the shared KV cache, and the loop continues with the next block.
- Reported by Orthrus: up to **7.8×** faster, training only **~16%** of parameters on **<1B** tokens.

> *These figures describe the Orthrus host framework (prior work), not FlowDraft's own results.*

## The problem

- Throughput of any verify-based system = **acceptance length** (drafted tokens accepted per cycle).
- Orthrus's drafter is a **single-step masked diffusion** model → it assumes block positions are conditionally independent → drafts diverge → tokens get rejected.
- Refining the draft would help, but **adding a step costs a forward pass** and lowers throughput.
- We need a **better proposal per pass**, not more passes.

## Key idea: a Categorical Flow Map drafter

- **Categorical Flow Maps** [Roos et al., 2026] learn the *integrated, correlated* endpoint distribution on the simplex and generate in **one or few jumps**.
- Use it as the drafter: a **higher-fidelity joint proposal** over the block — at the **same pass count**.
- Verification is unchanged → output stays **strictly lossless**; the drafter only affects *speed*, never *quality*.
- **Novelty:** a flow-map drafter inside Orthrus, trained with categorical VFM endpoint inference and flow-map consistency while retaining optional verifier alignment.

**Why it matters**

1. **Efficiency** — higher acceptance length = higher throughput, for free.
2. **Fidelity** — speedup with **zero** quality loss (verification guarantees it).
3. **Foundations** — connects flow-map distillation to fast, faithful LLM inference.

## CFM training, in brief

The drafter learns two complementary parts of a categorical flow map:

- **Endpoint inference — *what endpoint belongs to the trajectory*.** The diagonal predictor is trained by categorical VFM against the clean endpoint used to construct the interpolant.
- **Self-consistency — *how to jump*.** The reliable diagonal prediction at a transported state teaches the harder long-jump predictor through ECLD.

The ECLD target is stop-gradiented. An AR-verifier KL can be enabled as a separate auxiliary with `train.ar_kl_weight`, but defaults to zero because it is not part of the paper's CFM objective.

The AR model remains frozen throughout. In paper-faithful CFM training it supplies the cached prefix for the block-wise geometry and validation targets; at inference it verifies every proposal, which is what guarantees losslessness.

## Goals

1. **Reproduce Orthrus** (frozen AR + masked-diffusion drafter, shared KV cache, lossless loop) at a tractable scale.
2. **Implement a flow-map drafter** (simplex endpoint head, 1–few jumps).
3. **Train the categorical endpoint map** (VFM endpoint inference + flow-map consistency), with optional AR-KL alignment.
4. **Evaluate & compare:** AR baseline vs. masked-diffusion Orthrus vs. flow-map drafter — on acceptance length, TPF, and throughput — all verified lossless.

## Expected deliverables

1. Reproduction of the Orthrus lossless parallel decoder (masked-diffusion drafter).
2. Implementation of the **Categorical Flow Map drafter** + VFM/ECLD training.
3. Evaluation: acceptance-length / TPF / throughput comparison, with verified losslessness and **block-size / jump-count ablations**.

## Method

One frozen backbone, two attention paths (the Orthrus host), and a Categorical Flow Map drafter trained with VFM endpoint inference plus ECLD. Implemented; large-scale validation pending.

- **Adapter** (`src/models/base/df_adapter.py`): every `q/k/v_proj` gets a trainable twin initialized as a copy of the frozen AR weight (~14% of a 3B backbone). Routing is stateless (`torch.func.functional_call`, the backbone module tree is never modified); norms / MLP / `o_proj` / embeddings / LM head and one KV cache are shared. The cache is AR-only by contract: the drafter reads the committed prefix, its own K/V are cropped right after each forward. The DF path runs **unmasked** (bidirectional; CFM needs no attention mask beyond padding) and is conditioned on the jump times `(s, t)` via a zero-initialized sinusoidal time embedding (`fte.py`).
- **Objective** (`FlowDraft.compute_loss`): `loss = endpoint_weight·endpoint + ar_kl_weight·AR_KL + λ·(4·EC + 2·TD)`
  - **endpoint** — `CE(x1, π_{t,t}(x_t))`: the paper's categorical VFM diagonal objective. The paper-faithful evaluation point is `train.anchor_point=trajectory`; `landing` is retained as an experimental option.
  - **AR KL** — optional `KL(sg(p_AR) ‖ π_{t,t})`, separately weighted and off by default because it is not part of the CFM objective.
  - **EC** — eq. (18) of *Categorical Flow Maps*: `CE(sg(π_{t,t}(X_{s,t}(x_s))), π_{s,t}(x_s))` — jumps learn from the diagonal at their own landing point; truth flows `x1 → π_{t,t} → π_{s,t}`.
  - **TD** — eq. (16): temporal drift `‖∂_t π_{s,t}‖²`.
  - Time pairs `(s, t)` per sample (`train.time_sampling`): `paper` (default: t~U, s~U[0,t]) | `triangle` | `sequential`.
- **Training geometries** (`train.variant`): the task's variants are full-sequence — `flowdraft` (noise the whole sequence) and `orthrus` (Orthrus' own single-step masked-diffusion drafter: no time conditioning, barycenter as the simplex-native `[MASK]`). Our **addition beyond the task**: `flowdraft_block_wise` / `orthrus_block_wise` — the same two drafters retrained in the exact inference geometry (clean AR prefix in the KV cache, a CLEAN in-block anchor position — the decode loop's pending token, see below — and a noisy K-token block; also shrinks every `[B,T,V]` loss tensor to `[B,K,V]`).
- **Decoding** (`FlowDraft.generate`): draft K fresh tokens in 1–few jumps → ONE AR forward verifies the block. The previous cycle's correction/bonus token is never committed by its own pass: it rides as a clean in-block anchor and the next verify forward commits its K/V while scoring the drafts — **cycle cost = `jumps + 1` forwards** (TPF parity with the Orthrus convention). `temperature=0`: greedy verification, output **bit-identical** to `ar_generate`. `temperature>0` with Gumbel-coupled sampling (default): position-keyed Gumbel noise turns sampling into a deterministic argmax — the output is **bit-identical** to sampled `ar_generate` with the same seed. Uncoupled (`coupled=false`): Leviathan speculative sampling, lossless **in distribution**.

## Repository structure

```text
FlowDraft/
├── main.py                        # playground CLI (typer): generate from your prompts
├── hf-auth.sh                     # HF_TOKEN from .env -> env (gated Llama)
├── pyproject.toml                 # uv project; installed as an editable `src` package
└── src/
    ├── models/
    │   ├── base/df_adapter.py     # FlowDraftAttentionAdapter: frozen AR + trainable DF twins
    │   ├── base/fte.py            # FlowTimeEmbedding (s, t)
    │   ├── model.py               # build_model: backbone + tokenizer + processor
    │   ├── factory.py             # build_lit: variant selection + checkpoint loading
    │   ├── flowdraft.py           # FlowDraft: loss, training, lossless generate
    │   ├── flowdraft_block_wise.py        # FlowDraft in the inference geometry
    │   ├── orthrus.py             # Orthrus masked drafter (full-sequence)
    │   └── orthrus_block_wise.py          # Orthrus masked drafter, block-causal
    ├── preprocessor/df_processor.py   # tokenization + one-hot simplex endpoints
    ├── data/dataloaders.py        # streaming Dataset / collate / DataLoader;
    │                              #   EpochShuffled: repetitions in a new order (epochs)
    ├── configs/                   # hydra configs
    │   ├── train.yaml             # training entrypoint config
    │   ├── eval.yaml              # evaluation entrypoint config
    │   ├── model/                 # qwen3_1.7b (default) | qwen2_0.5b | llama3_3b
    │   ├── data/                  # nemotron (training) | math500 (eval, unseen in training)
    │   └── experiment/            # one preset per task stage + additions:
    │                              #   orthrus | flowdraft_staged | ablate_teacher_only |
    │                              #   ablate_consistency_only | orthrus_block_wise |
    │                              #   flowdraft_block_wise
    ├── train.py                   # training entrypoint
    ├── eval.py                    # dataset evaluation: acceptance / TPF / NLL -> results/eval.jsonl
    └── plots.py                   # report figures: frontier / TPF bars / TPF-vs-K
```

## Installation

```bash
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env     # gated meta-llama access
./hf-auth.sh                      # verify the token authenticates
```

## Usage

```bash
# generate from your prompts (greedy: bitwise-lossless check included)
./hf-auth.sh uv run python main.py -p "Once upon a time" -p "def main():"
# sampling — bit-exact vs AR too (Gumbel coupling is the default; --no-coupled = lossless in distribution)
./hf-auth.sh uv run python main.py -p "..." --temperature 0.8 --top-k 50 \
    --jumps 2 --checkpoint checkpoints/last.ckpt
```

## Training

Data: [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2),
streamed (no full download), category splits interleaved, `messages` rendered
with the tokenizer's chat template (`src/data/dataloaders.py`). Batch contract:
`input_ids [B,T]` + `attention_mask [B,T]`; the `[B,T,V]` simplex is built
on-device, never in the batch.

```bash
./hf-auth.sh uv run python src/train.py                            # FlowDraft (the task's recipe)
./hf-auth.sh uv run python src/train.py +experiment=orthrus       # task presets: orthrus |
                                                                   #   flowdraft_staged | ablate_*
./hf-auth.sh uv run python src/train.py train.variant=flowdraft_block_wise   # ADDITION: inference geometry
```

Variants: `flowdraft` is the project flow-map objective; `orthrus` is the
paper-style Orthrus recipe (frozen AR cache plus 256 isolated anchored masked
blocks); `flowdraft_block_wise` and `orthrus_block_wise` are the project’s single-block
inference-geometry variants.
Knobs live in `configs/train.yaml`: `lambda`/`endpoint_weight`/`ar_kl_weight`/`lambda_ramp_steps`
(VFM/ECLD balance + staging), `anchor_point`, `time_sampling`,
`block_size`/`min_prefix`, `val_decode_prompts` (val-time decode -> `val/tpf`
curves + checkpoint monitor), `early_stop_patience`, optimizer, Lightning
`trainer.*`. Checkpoints store the DF head + its Adam moments (~5 GB for 3B;
the frozen backbone is never written). The Russian
section carries the full per-experiment guide.

**Resume after interruption.** `last.ckpt` is updated every 1,000 optimizer
steps by default. Resume the full Lightning state—DF weights, AdamW, cosine
schedule, global step, and callbacks—with:

```bash
./hf-auth.sh uv run python src/train.py +experiment=orthrus \
    resume_from_checkpoint=checkpoints/orthrus/last.ckpt \
    trainer.accelerator=gpu trainer.devices=2 trainer.strategy=ddp \
    trainer.accumulate_grad_batches=64
```

**Repeating the stream (epochs).** The dataset streams, so an "epoch" is
whatever you define. Every new Trainer epoch re-opens the stream in a NEW
order (per-epoch reshuffle; the validation slice is split off before the
shuffle, so it never leaks into training). Two ways to bound a repetition:

```bash
# Orthrus paper preset: 600K packed sequences, 2 epochs, Qwen3-1.7B.
# On 8 GPUs it uses micro-batch 1 and accumulation 16 (global batch 128):
uv run python src/train.py +experiment=orthrus trainer.devices=8
# or bound by steps per repetition instead of samples:
uv run python src/train.py trainer.max_steps=-1 trainer.max_epochs=3 trainer.limit_train_batches=2000
```

Without `data.train_size` each repetition draws FRESH samples from the huge
stream (more diversity, not strict epochs) — with it, exactly the same pool
in a new order.

**LR schedule.** Default: linear warmup (5% of steps) then cosine decay to
zero — the peak is `train.lr`, the horizon is taken from `trainer.max_steps`
(or `limit_train_batches` × `max_epochs`), the current value is logged as the
`lr-AdamW` curve. `train.lr_schedule=constant` turns it off.

The Orthrus paper preset uses 2 epochs over 600K packed 2048-token sequences,
256 anchored masked blocks of size 32 per sequence, global batch 128, cosine
2e-4, and 5% warmup. For two GPUs, preserve the global batch with 64
accumulation steps:

```bash
./hf-auth.sh uv run python src/train.py +experiment=orthrus \
    trainer.accelerator=gpu trainer.devices=2 trainer.strategy=ddp \
    trainer.accumulate_grad_batches=64
```

(`max_steps=9375` = 600000 samples × 2 epochs / global batch 128.)

## Configuration reference

All configs live in `src/configs/` (hydra). Any key can be overridden from the
command line (`train.lr=3e-4`), config groups are swapped whole
(`model=qwen3_1.7b data=nemotron`), presets are added with `+experiment=...`.

**`train.yaml` — training (`src/train.py`)**

| Key | Default | What it does |
| --- | --- | --- |
| `seed` | 42 | global RNG seed: data shuffle, noise draws, init |
| `output_dir` | `checkpoints` | where checkpoints, TensorBoard logs, and local W&B data land |
| `wandb.enabled` | false | mirror all Lightning training/validation metrics to W&B |
| `wandb.project` / `entity` / `name` | `flowdraft` / null / null | W&B destination and optional run name; null uses W&B defaults |
| `wandb.group` / `tags` | null / [] | optional W&B organization metadata |
| `wandb.offline` | false | record locally for a later `wandb sync` instead of uploading live |
| `train.variant` | `flowdraft` | which drafter to train: the task — `flowdraft` \| `orthrus` (full-sequence); the addition — `flowdraft_block_wise` \| `orthrus_block_wise` (inference geometry) |
| `train.block_size` | 64 | K — block length seen in training (block-wise variants) |
| `train.min_prefix` | 1 | shortest clean prefix before the training block |
| `train.lr` / `weight_decay` / `betas` | 1e-4 / 0.01 / [0.9, 0.95] | AdamW over the DF head only; `lr` is the PEAK of the schedule |
| `train.lr_schedule` | `cosine` | `cosine` (linear warmup → cosine decay to 0; needs a finite `trainer.max_steps` or `limit_train_batches`+`max_epochs`) \| `constant` |
| `train.warmup_ratio` | 0.05 | cosine only: fraction of total steps spent warming up |
| `train.time_sampling` | `paper` | how (s, t) pairs are drawn: `paper` \| `triangle` \| `sequential` |
| `train.lambda` | 1.0 | weight of the consistency part (4·EC + 2·TD) |
| `train.endpoint_weight` | 1.0 | weight of categorical VFM endpoint CE; 0 = endpoint-off ablation |
| `train.ar_kl_weight` | 0.0 | optional AR-verifier KL auxiliary; 0 keeps the objective paper-faithful |
| `train.lambda_ramp_steps` | 0 | staged distillation: lambda 0 → `lambda` over N steps; 0 = static |
| `train.anchor_point` | `trajectory` | where the anchor evaluates the diagonal: `trajectory` = π_{t,t}(x_t) \| `landing` = π_{t,t}(X_{s,t}(x_s)) |
| `train.checkpoint_name` | `flowdraft-{step:07d}` | checkpoint filename pattern — set your own per experiment (quote on CLI: `'train.checkpoint_name="my-run-{step:07d}"'`) |
| `train.checkpoint_every_n_steps` | 1000 | how often to checkpoint |
| `train.val_decode_prompts` / `val_decode_max_new` | 2 / 32 | run the real decode loop on N val prompts each validation → `val/tpf`, `val/acceptance_decode`; 0 = off |
| `train.monitor` / `monitor_mode` | `val/tpf` / `max` | which curve selects the best checkpoint |
| `train.early_stop_patience` | 5 | stop after N validations without `val/loss` improvement; 0 = off |
| `trainer.*` | — | passed verbatim to `lightning.Trainer` (precision, max_steps, …) |

**`eval.yaml` — metrics on a dataset (`src/eval.py`)**

| Key | Default | What it does |
| --- | --- | --- |
| `checkpoint` | null | trained DF-head `.ckpt`; null = untrained drafter |
| `checkpoint_config` | true | restore the saved backbone/tokenizer/adapter config, train parameters, and variant |
| `variant` | null | inferred from checkpoint; without a checkpoint null selects `flowdraft` |
| `results_file` | `results/eval.jsonl` | every run appends one JSON row (input of `src/plots.py`) |
| `decode.block_size` / `decode.jumps` | 8 / 1 | inference-time K and refinement passes — knobs of EVERY variant; `block_size` is NOT related to the `flowdraft_block_wise` training variant (see the plain-words guide below) |
| `decode.max_new_tokens` | 64 | tokens generated per prompt |
| `decode.n_prompts` | 64 | prompts taken from the dataset (100–200 for a paper table) |
| `decode.prompt_len` | null | null = the full rendered prompt; int N = first N tokens only |
| `decode.temperature` / `top_k` / `top_p` | 0 / null / null | 0 = greedy; >0 = sampling |
| `decode.coupled` | true | T>0: Gumbel-coupled sampling — bit-exact vs AR |
| `decode.equiv_samples` | 0 | uncoupled only: N draws for the TV law-equivalence test; 0 = off |

**`model/*` — backbone** (`qwen3_1.7b` default; `qwen2_0.5b` and `llama3_3b` remain available):
`name` (HF id), `backbone.dtype`, `backbone.device_map`,
`backbone.attn_implementation` (`sdpa` default \| `flex_attention` GPU-only \| `eager`).

**`data/*` — dataset** (`nemotron` for training, `math500` — unseen during training — for eval):
`dataset` (HF id), `splits`, `text_field` (column for plain-text benches),
`streaming`, `shuffle_buffer`, `val_size` (first N stream samples → validation),
`train_size` (null = the whole stream; int N = a fixed pool of N samples, so
`trainer.max_epochs` repeats exactly them), `batch_size`, `max_length`, `num_workers`.

**`experiment/*` — one preset per task stage, plus the addition presets.**
Each sets its own `output_dir` and `train.checkpoint_name`, so runs never
overwrite each other:

| Preset | Sets | Checkpoints |
| --- | --- | --- |
| `orthrus` | `variant=orthrus` | `checkpoints/orthrus/orthrus-*.ckpt` |
| `flowdraft_staged` | `variant=flowdraft`, `lambda_ramp_steps=2000` | `checkpoints/flowdraft-staged/flowdraft-staged-*.ckpt` |
| `ablate_teacher_only` | `variant=flowdraft`, `lambda=0` (endpoint-only; legacy preset name) | `checkpoints/ablate-endpoint/ablate-endpoint-*.ckpt` |
| `ablate_consistency_only` | `variant=flowdraft`, `endpoint_weight=0` | `checkpoints/ablate-consistency/ablate-consistency-*.ckpt` |
| `orthrus_block_wise` (addition) | `variant=orthrus_block_wise` | `checkpoints/orthrus-block-wise/orthrus-block-wise-*.ckpt` |
| `flowdraft_block_wise` (addition) | `variant=flowdraft_block_wise`, `lambda_ramp_steps=2000` | `checkpoints/flowdraft-block-wise/flowdraft-block-wise-*.ckpt` |

Your own experiment (e.g. the `anchor_point` study) — override name and dir
so it gets its own shelf too:

```bash
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_staged \
    train.anchor_point=landing \
    output_dir=checkpoints/anchor-landing 'train.checkpoint_name="anchor-landing-{step:07d}"'
```

## Inference parameters, in plain words

One decode cycle works like this: the drafter guesses a whole block of tokens
at once, the frozen base model checks the guess in a single pass, the leading
tokens that match what the base model would have said are kept, and the base
model adds one token of its own (the fix for the first wrong guess — or a
bonus token if everything matched). Then the next cycle starts. The knobs:

- `--block-size` (K) — how many tokens the drafter guesses per cycle. Bigger
  blocks promise more speedup, but the tail of a long guess relies on the
  guessed (unverified) beginning, so it gets rejected more often. Sweep 4–16.
  Despite the similar name this has nothing to do with the `flowdraft_block_wise`
  training variant — every drafter proposes blocks at inference.
- `--jumps` — how many passes the drafter spends polishing its guess before
  showing it to the base model. Each extra pass makes the guess better but
  costs one forward: a cycle costs `jumps + 1` passes total. More jumps only
  pay off if the extra accepted tokens outweigh the extra passes.
- `--max-new-tokens` — response length cap.
- `--temperature` — 0: always take the most likely token; the output is
  guaranteed identical to the plain base model, checked bit-for-bit. Above 0:
  random sampling, livelier text.
- `--top-k` / `--top-p` — sampling only: limit the draw to the k most likely
  tokens / the smallest set covering probability p.
- `--coupled` (on by default) — when sampling, the drafter and the base model
  draw their randomness from one shared, seeded source. Result: even the
  *sampled* text is exactly the text the plain base model would produce with
  that seed — token for token. `--sampling-seed` picks which text that is;
  `--no-coupled` switches to classic speculative sampling (same distribution,
  not the same tokens).
- `--variant` + `--checkpoint` — which drafter geometry to load and its
  trained weights. Without a checkpoint the drafter is untrained: output is
  still exact, it just accepts almost nothing (slow).
- `--model` — the backbone: a config name (`qwen2_0.5b`) or an HF id.

None of these affect *what* is generated beyond the guarantees above — only
how fast. The verifier has the final word on every token.

## Evaluation

Dataset prompts (full rendered prompts by default; `decode.prompt_len=N` for
N-token prefixes) are decoded twice — flow-draft vs plain AR — and compared. Greedy losslessness is asserted **bitwise**, not assumed.

When `checkpoint` is set, evaluation resolves the path from the original
working directory, restores the saved model architecture and variant, and
strictly loads every trainable DF tensor. Missing, unknown, or shape-mismatched
parameters fail before evaluation instead of being silently ignored. Runtime
device, dtype, attention-kernel, and compile settings remain controlled by the
evaluation config. Use `checkpoint_config=false` only for a legacy checkpoint
without metadata, together with explicit matching `model=... variant=...`.

```bash
./hf-auth.sh uv run python src/eval.py checkpoint=path.ckpt   # variant=flowdraft is the default
# block-size / jump-count ablation grid (hydra multirun):
./hf-auth.sh uv run python src/eval.py -m decode.block_size=4,8,16 decode.jumps=1,2,4
```

Main metrics (mean ± std over `n_prompts`): **acceptance** per cycle and
**TPF** (tokens per forward; cycle = `jumps+1`). Wall-clock tokens/s and
speedup vs AR are reported as diagnostics (hardware/kernel dependent). The
attention kernel is a config switch (`model.backbone.attn_implementation`):
`sdpa` (default; fused, supports the DF mask — verified against eager) |
`flex_attention` (compiled block masks, GPU only) | `eager` (reference).
**Continuation NLL** under the frozen teacher is computed in sampling mode
only (at greedy the output is bitwise equal to AR, so it measures nothing).

The Orthrus quality table covers five benchmark families: **GSM8K**,
**MATH-500**, **AIME**, **HumanEval**, and **MBPP**. AIME is represented by
separate 2024 and 2025 sets, so the runnable suite has six dataset configs:
`gsm8k`, `math500`, `aime24`, `aime25`, `humaneval`, and `mbpp`. The broader
Orthrus efficiency table additionally reports Pseudo2Code and
LiveCodeBench-v5. `data=nemotron` remains available for measuring the gap to
the training distribution.

Run the paper-style greedy, K=32 protocol once for FlowDraft and once for the
Orthrus baseline (use the checkpoint belonging to each variant):

```bash
./hf-auth.sh uv run python src/eval.py -m +benchmark=orthrus \
    data=gsm8k,math500,aime24,aime25,humaneval,mbpp \
    variant=flowdraft checkpoint=/absolute/path/flowdraft.ckpt
./hf-auth.sh uv run python src/eval.py -m +benchmark=orthrus \
    data=gsm8k,math500,aime24,aime25,humaneval,mbpp \
    variant=orthrus checkpoint=/absolute/path/orthrus.ckpt
```

Every prompt is decoded by the selected drafter and by plain AR; bitwise
identity is asserted before acceptance, TPF, and throughput are reported.
Consequently benchmark quality is inherited exactly from the frozen AR model;
HumanEval/MBPP functional pass rates still require their official sandboxed
code-execution harnesses.
Bench problems are wrapped with the verifier's chat template (user turn +
generation prompt, with Qwen3 thinking disabled as in Orthrus) and decoded from the **full prompt**
(`decode.prompt_len=null`); set an int for prefix-continuation mode.

## Results

> 🚧 **TODO.** Fill in once experiments are done. All rows must be verified lossless.

| Method | Acceptance length ↑ | TPF * | Throughput (tok/s) ↑ | Lossless |
| --- | --- | --- | --- | --- |
| AR baseline | — | — | — | ✅ (trivially) |
| Orthrus (masked-diffusion drafter) | TBD | TBD | TBD | ✅ |
| **FlowDraft** (flow-map drafter) | TBD | TBD | TBD | ✅ |

\* *TPF — tokens per forward pass: `N generated / N forwards`, one cycle = `jumps + 1` forwards (formulas: the [Russian guide](README.ru.md)).*

**Ablations (TODO):** block size, jump count.

## References

- **Categorical Flow Maps** — Roos et al., ICML 2026. arXiv:2602.12233. Reference implementation: `olsdavis/semicat`. <!-- TODO: confirm final citation & links -->
- **Orthrus** — lossless parallel decoding via a frozen AR backbone + trainable diffusion drafter. arXiv:2605.12825. Reference implementation: `chiennv2000/orthrus`. <!-- TODO: confirm final citation & links -->

<!-- TODO: complete once metadata is available -->
```bibtex
@misc{flowdraft2026,
  title  = {FlowDraft: Flow-Map Drafting for Lossless Parallel Decoding},
  author = {TODO},
  year   = {2026},
  note   = {Summer of Machine Learning at Skoltech (SMILES), Applied AI Center}
}
```

## Team

- **Contributors:** <!-- TODO: team members -->
- **Curators / mentors:** Maria Ivanova (YSDA, Applied AI Institute) · Dmitrii Babaev

## Acknowledgments

Developed as part of the **Summer of Machine Learning at Skoltech (SMILES)**, Skoltech Applied AI Center.

## License

> 🚧 **TODO:** choose and add a license (e.g., MIT / Apache-2.0).
