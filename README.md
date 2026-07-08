# FlowDraft: Flow-Map Drafting for Lossless Parallel Decoding

> Raising the **acceptance ceiling** of lossless parallel decoding by upgrading the *drafter* to a **Categorical Flow Map** — faster generation, provably identical output.

**Language / Язык:** [English](#english) · [Русский](#русский)

<!-- Badges — TODO: fill in once the repo is public
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-WIP-orange)
-->

> 🚧 **Status: core implementation landed and smoke-verified** — adapter, four training variants (the task's full-sequence fixed / baseline + our block-wise additions; trained end-to-end on real data: SmolLM2-135M + Nemotron), lossless decoding (**bitwise** at greedy AND at sampling via Gumbel coupling; `jumps+1` forwards per cycle), evaluation harness (mean±std, JSONL results + report plots), experiment presets for every stage of the task. GPU experiments pending; **Results** are TBD.

**Summer of Machine Learning at Skoltech (SMILES) · Applied AI Center**

---

## English

### Table of contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Experiments (task stages)](#experiments-task-stages)
- [Background: the decoding bottleneck](#background-the-decoding-bottleneck)
- [Host framework: Orthrus](#host-framework-orthrus)
- [The problem](#the-problem)
- [Key idea: a Categorical Flow Map drafter](#key-idea-a-categorical-flow-map-drafter)
- [Dual distillation, in brief](#dual-distillation-in-brief)
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

### Overview

Autoregressive (AR) LLMs decode strictly sequentially: generating *L* tokens costs *L* forward passes, which is memory-bandwidth bound. Diffusion LMs can draft whole blocks in parallel, but they drift from the AR distribution and lose quality. Speculative-style verification restores quality: draft a block in parallel, then verify it against the AR model in a single pass and keep only the tokens the AR model would have produced — this is **lossless**.

**FlowDraft** upgrades the *drafter* inside a lossless parallel-decoding loop. The throughput of any verify-based system is governed by its **acceptance length** — the number of drafted tokens accepted per cycle. We replace the single-step masked-diffusion drafter with a **Categorical Flow Map** drafter that produces a higher-fidelity *joint* proposal over the block at the **same** number of forward passes. Verification is left untouched, so the output stays strictly lossless — the drafter affects only **speed**, never **quality**.

Crucially, the AR model is what does the verifying, so it is kept **frozen throughout**. Keeping it untouched is exactly what makes the output provably identical to the base model; it is what the word *lossless* rests on.

### Quickstart

```bash
# 1. Setup (once)
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env        # gated meta-llama access
./hf-auth.sh                         # verify: prints your HF username

# 2. Check inference works — the UNTRAINED drafter is already lossless (just slow)
./hf-auth.sh uv run python main.py -p "Once upon a time"
#    -> generation + [lossless vs greedy AR: PASS]

# 3. Train the drafter (fixed = the task's full-sequence recipe; GPU recommended)
./hf-auth.sh uv run python src/train.py \
    trainer.max_steps=10000 data.batch_size=8
#    watch: loss/anchor ↓, loss/ec ↓, loss/td sane, val/teacher_agreement ↑
#    checkpoints (DF head + Adam moments, ~5 GB for 3B) land in checkpoints/
#    our ADDITION beyond the task — training in the exact inference geometry:
#    append train.variant=block_wise
#    epochs on top of streaming (nothing is downloaded ahead): a fixed pool of
#    N samples repeated M times, each repetition in a new order —
#    ./hf-auth.sh uv run python src/train.py \
#        data.train_size=471952 trainer.max_epochs=2 trainer.max_steps=7375

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

### Experiments (task stages)

Every stage of the project task is one preset in `src/configs/experiment/`. The
**detailed walkthrough — what each stage means, which training curves to
watch, expected behaviour, results-table rows, and the analysis pipeline —
is the Russian guide: [Эксперименты (этапы задания)](#эксперименты-этапы-задания).**
The command summary:

```bash
# Stage 1 — reproduce the Orthrus masked-diffusion baseline @ 0.5B
./hf-auth.sh uv run python src/train.py +experiment=baseline

# Stages 2-3 — flow-map drafter, staged dual distillation (teacher first, consistency ramped in)
./hf-auth.sh uv run python src/train.py +experiment=flowmap_staged

# Stage 4 — lossless at sampling: coupled = bitwise; uncoupled = null-calibrated TV test
./hf-auth.sh uv run python src/eval.py model=qwen2_0.5b checkpoint=<ckpt> decode.temperature=0.8
./hf-auth.sh uv run python src/eval.py model=qwen2_0.5b checkpoint=<ckpt> decode.temperature=0.8 \
    decode.coupled=false decode.equiv_samples=500

# Stage 5 (ablations) — the contribution of each distillation term
./hf-auth.sh uv run python src/train.py +experiment=ablate_teacher_only
./hf-auth.sh uv run python src/train.py +experiment=ablate_consistency_only

# Stage 5 (evaluation) — default: MATH-500, a dataset never seen in training
#           (data=nemotron evaluates on the training distribution);
#           block-size x jumps grid -> results/eval.jsonl -> report figures.
#           NOTE: decode.block_size = K, the number of tokens drafted per cycle at
#           INFERENCE — a knob of EVERY variant (fixed included); it is unrelated
#           to the block_wise TRAINING-geometry variant despite the similar name
./hf-auth.sh uv run python src/eval.py -m model=qwen2_0.5b variant=fixed checkpoint=<ckpt> \
    decode.block_size=4,8,16 decode.jumps=1,2,4
uv run python src/plots.py

# ADDITION (beyond the task) — both drafters retrained in the exact inference
# geometry (block-causal): the geometry-for-geometry comparison
./hf-auth.sh uv run python src/train.py +experiment=baseline_block_wise
./hf-auth.sh uv run python src/train.py +experiment=flowmap_block_wise
#   eval: same stage-5 commands with variant=baseline_block_wise / block_wise
```

Training curves land in TensorBoard (`uv run tensorboard --logdir checkpoints`);
metric formulas and the training↔eval correspondence are spelled out in the
Russian guide.

### Background: the decoding bottleneck

- **AR LLMs** decode strictly sequentially: *L* tokens → *L* forward passes (memory-bandwidth bound).
- **Diffusion LMs** draft blocks in parallel, but drift from the AR distribution and lose quality.
- **Speculative-style verification** fixes quality: draft in parallel, then *verify* against the AR model → keep only correct tokens (**lossless**).

### Host framework: Orthrus

FlowDraft is built inside **Orthrus**, a lossless parallel-decoding scaffold:

- One transformer, two attention paths: a **frozen AR path** and a **lightweight, trainable diffusion path** (~16% of parameters), sharing the same norm / MLP / embeddings and a single KV cache.
- The diffusion path proposes *K* tokens in parallel; the frozen AR head verifies them in one pass → output **provably identical** to the base model. Accepted tokens are committed to the shared KV cache, and the loop continues with the next block.
- Reported by Orthrus: up to **7.8×** faster, training only **~16%** of parameters on **<1B** tokens.

> *These figures describe the Orthrus host framework (prior work), not FlowDraft's own results.*

### The problem

- Throughput of any verify-based system = **acceptance length** (drafted tokens accepted per cycle).
- Orthrus's drafter is a **single-step masked diffusion** model → it assumes block positions are conditionally independent → drafts diverge → tokens get rejected.
- Refining the draft would help, but **adding a step costs a forward pass** and lowers throughput.
- We need a **better proposal per pass**, not more passes.

### Key idea: a Categorical Flow Map drafter

- **Categorical Flow Maps** [Roos et al., 2026] learn the *integrated, correlated* endpoint distribution on the simplex and generate in **one or few jumps**.
- Use it as the drafter: a **higher-fidelity joint proposal** over the block — at the **same pass count**.
- Verification is unchanged → output stays **strictly lossless**; the drafter only affects *speed*, never *quality*.
- **Novelty:** a flow-map drafter + a **dual distillation** that fits both the AR-teacher distribution **and** flow-map consistency.

**Why it matters**

1. **Efficiency** — higher acceptance length = higher throughput, for free.
2. **Fidelity** — speedup with **zero** quality loss (verification guarantees it).
3. **Foundations** — connects flow-map distillation to fast, faithful LLM inference.

### Dual distillation, in brief

The drafter is trained with **two teachers** — hence *dual* — because a single teacher cannot supply both skills it needs:

- **From the AR model — *what* to propose.** The drafter learns to match the distribution the frozen AR verifier would accept, rather than to reproduce the training corpus. The target is to be *accepted by the verifier*, not to match corpus text.
- **From itself — *how* to jump.** The drafter's reliable "local" behaviour teaches its harder "long-jump" behaviour, so it can emit the whole block in one or a few jumps. This is the flow-map (self-)consistency.

Both teachers are used as fixed targets (stop-gradient) — the model is not trained *through* them. The balance between the two terms is the main design knob.

The frozen AR model is used **only as a teacher** during training: it provides the target distribution, and its weights are never updated. It stays frozen at inference too, where it is the verifier — the same fact that guarantees losslessness.

### Goals

1. **Reproduce Orthrus** (frozen AR + masked-diffusion drafter, shared KV cache, lossless loop) at a tractable scale.
2. **Implement a flow-map drafter** (simplex endpoint head, 1–few jumps).
3. **Develop the dual-distillation objective** (AR-teacher distribution + flow-map consistency).
4. **Evaluate & compare:** AR baseline vs. masked-diffusion Orthrus vs. flow-map drafter — on acceptance length, TPF, and throughput — all verified lossless.

### Expected deliverables

1. Reproduction of the Orthrus lossless parallel decoder (masked-diffusion drafter).
2. Implementation of the **Categorical Flow Map drafter** + dual-distillation training.
3. Evaluation: acceptance-length / TPF / throughput comparison, with verified losslessness and **block-size / jump-count ablations**.

### Method

One frozen backbone, two attention paths (the Orthrus host), a Categorical Flow Map drafter on the diffusion path, and a dual-distillation objective. Implemented; large-scale validation pending.

- **Adapter** (`src/models/base/df_adapter.py`): every `q/k/v_proj` gets a trainable twin initialized as a copy of the frozen AR weight (~14% of a 3B backbone). Routing is stateless (`torch.func.functional_call`, the backbone module tree is never modified); norms / MLP / `o_proj` / embeddings / LM head and one KV cache are shared. The cache is AR-only by contract: the drafter reads the committed prefix, its own K/V are cropped right after each forward. The DF path runs **unmasked** (bidirectional; CFM needs no attention mask beyond padding) and is conditioned on the jump times `(s, t)` via a zero-initialized sinusoidal time embedding (`fte.py`).
- **Objective** (`FlowMapOrthrus.compute_loss`): `loss = anchor + λ · (4·EC + 2·TD)`
  - **anchor** — `KL(sg(p_AR) ‖ π_{t,t}(·))`: the AR verifier anchors only the diagonal (soft target). The evaluation point is a knob (`train.anchor_point`): trajectory `x_t` (default) | landing `X_{s,t}(x_s)`. Jumps get no direct AR loss — such a target is constant in the noise seed and collapses the transport.
  - **EC** — eq. (18) of *Categorical Flow Maps*: `CE(sg(π_{t,t}(X_{s,t}(x_s))), π_{s,t}(x_s))` — jumps learn from the diagonal at their own landing point; truth flows `p_AR → π_{t,t} → π_{s,t}`.
  - **TD** — eq. (16): temporal drift `‖∂_t π_{s,t}‖²`.
  - Time pairs `(s, t)` per sample (`train.time_sampling`): `triangle` (uniform on {s≤t}) | `sequential` | `paper` (t~U, s~U[0,t]).
- **Training geometries** (`train.variant`): the task's variants are full-sequence — `fixed` (noise the whole sequence) and `baseline` (Orthrus' own single-step masked-diffusion drafter: no time conditioning, barycenter as the simplex-native `[MASK]`). Our **addition beyond the task**: `block_wise` / `baseline_block_wise` — the same two drafters retrained in the exact inference geometry (clean AR prefix in the KV cache, a CLEAN in-block anchor position — the decode loop's pending token, see below — and a noisy K-token block; also shrinks every `[B,T,V]` loss tensor to `[B,K,V]`).
- **Decoding** (`FlowMapOrthrus.generate`): draft K fresh tokens in 1–few jumps → ONE AR forward verifies the block. The previous cycle's correction/bonus token is never committed by its own pass: it rides as a clean in-block anchor and the next verify forward commits its K/V while scoring the drafts — **cycle cost = `jumps + 1` forwards** (TPF parity with the Orthrus convention). `temperature=0`: greedy verification, output **bit-identical** to `ar_generate`. `temperature>0` with Gumbel-coupled sampling (default): position-keyed Gumbel noise turns sampling into a deterministic argmax — the output is **bit-identical** to sampled `ar_generate` with the same seed. Uncoupled (`coupled=false`): Leviathan speculative sampling, lossless **in distribution**.

### Repository structure

```text
FlowDraft/
├── main.py                        # playground CLI (typer): generate from your prompts
├── hf-auth.sh                     # HF_TOKEN from .env -> env (gated Llama)
├── pyproject.toml                 # uv project; installed as an editable `src` package
└── src/
    ├── models/
    │   ├── base/df_adapter.py     # OrthrusAttentionAdapter: frozen AR + trainable DF twins
    │   ├── base/fte.py            # FlowTimeEmbedding (s, t)
    │   ├── model.py               # build_model: backbone + tokenizer + processor
    │   ├── factory.py             # build_lit: variant selection + checkpoint loading
    │   ├── lit_orthrus.py         # FlowMapOrthrus: loss, training, lossless generate
    │   ├── lit_orthrus_block_wise.py      # ADDITION: training in the inference geometry
    │   ├── lit_orthrus_baseline.py        # Orthrus masked drafter (full-sequence)
    │   └── lit_orthrus_baseline_block_wise.py  # ADDITION: masked drafter, block-causal
    ├── preprocessor/df_processor.py   # tokenization + one-hot simplex endpoints
    ├── data/dataloaders.py        # streaming Dataset / collate / DataLoader;
    │                              #   EpochShuffled: repetitions in a new order (epochs)
    ├── configs/                   # hydra configs
    │   ├── train.yaml             # training entrypoint config
    │   ├── eval.yaml              # evaluation entrypoint config
    │   ├── model/                 # llama3_3b (default) | qwen2_0.5b (the task's 0.5B)
    │   ├── data/                  # nemotron (training) | math500 (eval, unseen in training)
    │   └── experiment/            # one preset per task stage + additions:
    │                              #   baseline | flowmap_staged | ablate_teacher_only |
    │                              #   ablate_consistency_only | baseline_block_wise |
    │                              #   flowmap_block_wise
    ├── train.py                   # training entrypoint
    ├── eval.py                    # dataset evaluation: acceptance / TPF / NLL -> results/eval.jsonl
    └── plots.py                   # report figures: frontier / TPF bars / TPF-vs-K
```

### Installation

```bash
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env     # gated meta-llama access
./hf-auth.sh                      # verify the token authenticates
```

### Usage

```bash
# generate from your prompts (greedy: bitwise-lossless check included)
./hf-auth.sh uv run python main.py -p "Once upon a time" -p "def main():"
# sampling — bit-exact vs AR too (Gumbel coupling is the default; --no-coupled = lossless in distribution)
./hf-auth.sh uv run python main.py -p "..." --temperature 0.8 --top-k 50 \
    --jumps 2 --checkpoint checkpoints/last.ckpt
```

### Training

Data: [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2),
streamed (no full download), category splits interleaved, `messages` rendered
with the tokenizer's chat template (`src/data/dataloaders.py`). Batch contract:
`input_ids [B,T]` + `attention_mask [B,T]`; the `[B,T,V]` simplex is built
on-device, never in the batch.

```bash
./hf-auth.sh uv run python src/train.py                            # fixed (the task's recipe)
./hf-auth.sh uv run python src/train.py +experiment=baseline       # task presets: baseline |
                                                                   #   flowmap_staged | ablate_*
./hf-auth.sh uv run python src/train.py train.variant=block_wise   # ADDITION: inference geometry
```

Variants — the task: `fixed` | `baseline` (full-sequence); our addition:
`block_wise` | `baseline_block_wise` (the inference geometry).
Knobs live in `configs/train.yaml`: `lambda`/`anchor_weight`/`lambda_ramp_steps`
(dual distillation + staging), `anchor_point`, `time_sampling`,
`block_size`/`min_prefix`, `val_decode_prompts` (val-time decode -> `val/tpf`
curves + checkpoint monitor), `early_stop_patience`, optimizer, Lightning
`trainer.*`. Checkpoints store the DF head + its Adam moments (~5 GB for 3B;
the frozen backbone is never written). The Russian
section carries the full per-experiment guide.

**Repeating the stream (epochs).** The dataset streams, so an "epoch" is
whatever you define. Every new Trainer epoch re-opens the stream in a NEW
order (per-epoch reshuffle; the validation slice is split off before the
shuffle, so it never leaks into training). Two ways to bound a repetition:

```bash
# fixed sample pool, repeated N times — the paper-style "K examples x N epochs":
uv run python src/train.py trainer.max_steps=-1 trainer.max_epochs=2 data.train_size=471952
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

Putting it together — the training budget of the Orthrus paper (2 epochs over
472K packed sequences ≈ 1.9B tokens, global batch 128 × 2048 tokens, cosine
2e-4 with 5% warmup):

```bash
./hf-auth.sh uv run python src/train.py +experiment=baseline \
    data.train_size=471952 data.batch_size=8 data.max_length=2048 \
    trainer.accumulate_grad_batches=16 train.lr=2e-4 \
    trainer.max_epochs=2 trainer.max_steps=7375
```

(`max_steps` = 471952 samples / (8 × 16) per optimizer step × 2 epochs; the
cosine schedule needs it explicitly — a streaming loader has no length.)

### Configuration reference

All configs live in `src/configs/` (hydra). Any key can be overridden from the
command line (`train.lr=3e-4`), config groups are swapped whole
(`model=qwen2_0.5b data=nemotron`), presets are added with `+experiment=...`.

**`train.yaml` — training (`src/train.py`)**

| Key | Default | What it does |
| --- | --- | --- |
| `seed` | 42 | global RNG seed: data shuffle, noise draws, init |
| `output_dir` | `checkpoints` | where checkpoints and TensorBoard logs land |
| `train.variant` | `fixed` | which drafter to train: the task — `fixed` \| `baseline` (full-sequence); the addition — `block_wise` \| `baseline_block_wise` (inference geometry) |
| `train.block_size` | 64 | K — block length seen in training (block-wise variants) |
| `train.min_prefix` | 1 | shortest clean prefix before the training block |
| `train.lr` / `weight_decay` / `betas` | 1e-4 / 0.01 / [0.9, 0.95] | AdamW over the DF head only; `lr` is the PEAK of the schedule |
| `train.lr_schedule` | `cosine` | `cosine` (linear warmup → cosine decay to 0; needs a finite `trainer.max_steps` or `limit_train_batches`+`max_epochs`) \| `constant` |
| `train.warmup_ratio` | 0.05 | cosine only: fraction of total steps spent warming up |
| `train.time_sampling` | `triangle` | how (s, t) pairs are drawn: `triangle` \| `sequential` \| `paper` |
| `train.lambda` | 1.0 | weight of the consistency part (4·EC + 2·TD) |
| `train.anchor_weight` | 1.0 | weight of the AR-teacher anchor; 0 = teacher-off ablation |
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
| `variant` | `fixed` | must match how the checkpoint was trained |
| `results_file` | `results/eval.jsonl` | every run appends one JSON row (input of `src/plots.py`) |
| `decode.block_size` / `decode.jumps` | 8 / 1 | inference-time K and refinement passes — knobs of EVERY variant; `block_size` is NOT related to the `block_wise` training variant (see the plain-words guide below) |
| `decode.max_new_tokens` | 64 | tokens generated per prompt |
| `decode.n_prompts` | 64 | prompts taken from the dataset (100–200 for a paper table) |
| `decode.prompt_len` | null | null = the full rendered prompt; int N = first N tokens only |
| `decode.temperature` / `top_k` / `top_p` | 0 / null / null | 0 = greedy; >0 = sampling |
| `decode.coupled` | true | T>0: Gumbel-coupled sampling — bit-exact vs AR |
| `decode.equiv_samples` | 0 | uncoupled only: N draws for the TV law-equivalence test; 0 = off |

**`model/*` — backbone** (`llama3_3b` default, `qwen2_0.5b` — the 0.5B scale the task asks for):
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
| `baseline` | `variant=baseline` | `checkpoints/baseline/baseline-*.ckpt` |
| `flowmap_staged` | `variant=fixed`, `lambda_ramp_steps=2000` | `checkpoints/flowmap-staged/flowmap-staged-*.ckpt` |
| `ablate_teacher_only` | `variant=fixed`, `lambda=0` | `checkpoints/ablate-teacher/ablate-teacher-*.ckpt` |
| `ablate_consistency_only` | `variant=fixed`, `anchor_weight=0` | `checkpoints/ablate-consistency/ablate-consistency-*.ckpt` |
| `baseline_block_wise` (addition) | `variant=baseline_block_wise` | `checkpoints/baseline-block-wise/baseline-block-wise-*.ckpt` |
| `flowmap_block_wise` (addition) | `variant=block_wise`, `lambda_ramp_steps=2000` | `checkpoints/flowmap-block-wise/flowmap-block-wise-*.ckpt` |

Your own experiment (e.g. the `anchor_point` study) — override name and dir
so it gets its own shelf too:

```bash
./hf-auth.sh uv run python src/train.py +experiment=flowmap_staged \
    train.anchor_point=landing \
    output_dir=checkpoints/anchor-landing 'train.checkpoint_name="anchor-landing-{step:07d}"'
```

### Inference parameters, in plain words

One decode cycle works like this: the drafter guesses a whole block of tokens
at once, the frozen base model checks the guess in a single pass, the leading
tokens that match what the base model would have said are kept, and the base
model adds one token of its own (the fix for the first wrong guess — or a
bonus token if everything matched). Then the next cycle starts. The knobs:

- `--block-size` (K) — how many tokens the drafter guesses per cycle. Bigger
  blocks promise more speedup, but the tail of a long guess relies on the
  guessed (unverified) beginning, so it gets rejected more often. Sweep 4–16.
  Despite the similar name this has nothing to do with the `block_wise`
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

### Evaluation

Dataset prompts (full rendered prompts by default; `decode.prompt_len=N` for
N-token prefixes) are decoded twice — flow-draft vs plain AR — and compared. Greedy losslessness is asserted **bitwise**, not assumed.

```bash
./hf-auth.sh uv run python src/eval.py checkpoint=path.ckpt   # variant=fixed is the default
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

Two evaluation datasets (`configs/data/`): the **default is MATH-500**
(`data=math500`) — a dataset the drafter never saw during training, so its
acceptance/TPF is the honest generalization number; `data=nemotron` evaluates
on the training distribution (its validation slice is excluded from training,
but the texts are of the same kind). Report both: the gap between them shows
how much the drafter overfits the training mix.
Bench problems are wrapped with the verifier's chat template (user turn +
generation prompt) and decoded from the **full prompt**
(`decode.prompt_len=null`); set an int for prefix-continuation mode.

### Results

> 🚧 **TODO.** Fill in once experiments are done. All rows must be verified lossless.

| Method | Acceptance length ↑ | TPF * | Throughput (tok/s) ↑ | Lossless |
| --- | --- | --- | --- | --- |
| AR baseline | — | — | — | ✅ (trivially) |
| Orthrus (masked-diffusion drafter) | TBD | TBD | TBD | ✅ |
| **FlowDraft** (flow-map drafter) | TBD | TBD | TBD | ✅ |

\* *TPF — tokens per forward pass: `N generated / N forwards`, one cycle = `jumps + 1` forwards (formulas: the Russian guide).*

**Ablations (TODO):** block size, jump count.

### References

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

### Team

- **Contributors:** <!-- TODO: team members -->
- **Curators / mentors:** Maria Ivanova (YSDA, Applied AI Institute) · Dmitrii Babaev

### Acknowledgments

Developed as part of the **Summer of Machine Learning at Skoltech (SMILES)**, Skoltech Applied AI Center.

### License

> 🚧 **TODO:** choose and add a license (e.g., MIT / Apache-2.0).

---

## Русский

### Содержание

- [Обзор](#обзор)
- [Быстрый старт](#быстрый-старт)
- [Эксперименты (этапы задания)](#эксперименты-этапы-задания)
- [Контекст: узкое место декодирования](#контекст-узкое-место-декодирования)
- [Host-фреймворк: Orthrus](#host-фреймворк-orthrus)
- [Проблема](#проблема)
- [Идея: драфтер на Categorical Flow Map](#идея-драфтер-на-categorical-flow-map)
- [Коротко про dual distillation](#коротко-про-dual-distillation)
- [Цели](#цели)
- [Ожидаемые результаты](#ожидаемые-результаты)
- [Метод](#метод) 🚧
- [Структура репозитория](#структура-репозитория)
- [Установка](#установка) 🚧
- [Использование](#использование) 🚧
- [Обучение](#обучение) 🚧
- [Справочник конфигов](#справочник-конфигов)
- [Параметры инференса простыми словами](#параметры-инференса-простыми-словами)
- [Оценка](#оценка) 🚧
- [Результаты](#результаты) 🚧
- [Ссылки](#ссылки)
- [Команда](#команда)
- [Благодарности](#благодарности)
- [Лицензия](#лицензия) 🚧

### Обзор

Авторегрессионные (AR) LLM декодируют строго последовательно: чтобы сгенерировать *L* токенов, нужно *L* прямых проходов — а это упирается в пропускную способность памяти. Диффузионные LM умеют драфтить целые блоки параллельно, но уходят от AR-распределения и теряют качество. Верификация в стиле спекулятивного декодирования возвращает качество: драфтим блок параллельно, затем сверяем его с AR-моделью за один проход и оставляем только те токены, которые выдала бы сама AR-модель — это **без потерь (lossless)**.

**FlowDraft** улучшает *драфтер* внутри lossless-петли параллельного декодирования. Пропускную способность любой системы с верификацией определяет **длина приёма (acceptance length)** — сколько сдрафченных токенов принимается за цикл. Мы заменяем одношаговый masked-diffusion драфтер на драфтер на основе **Categorical Flow Map**, который выдаёт более качественное *совместное* предложение по блоку при **том же** числе прямых проходов. Верификация не меняется, поэтому вывод остаётся строго lossless — драфтер влияет только на **скорость**, но не на **качество**.

Важно: верифицирует именно AR-модель, поэтому она остаётся **замороженной на всех этапах**. Именно то, что её не трогают, и делает вывод доказуемо идентичным базовой модели — на этом держится слово *lossless*.

### Быстрый старт

```bash
# 1. Установка (один раз)
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env        # доступ к gated meta-llama
./hf-auth.sh                         # проверка: печатает ваш HF-логин

# 2. Проверка, что инференс работает — НЕОБУЧЕННЫЙ драфтер уже lossless (просто медленный)
./hf-auth.sh uv run python main.py -p "Once upon a time"
#    -> генерация + [lossless vs greedy AR: PASS]

# 3. Обучение драфтера (fixed = полноследовательный рецепт из задания; нужен GPU)
./hf-auth.sh uv run python src/train.py \
    trainer.max_steps=10000 data.batch_size=8
#    смотреть: loss/anchor ↓, loss/ec ↓, loss/td без пиков, val/teacher_agreement ↑
#    чекпоинты (DF-голова + Adam-моменты, ~5 ГБ для 3B) падают в checkpoints/
#    наше ДОПОЛНЕНИЕ сверх задания — обучение в точной инференсной геометрии:
#    добавьте train.variant=block_wise
#    эпохи поверх стриминга (ничего не скачивается заранее): фиксированный пул
#    из N сэмплов, повторённый M раз, каждый повтор в новом порядке —
#    ./hf-auth.sh uv run python src/train.py \
#        data.train_size=471952 trainer.max_epochs=2 trainer.max_steps=7375

# 4. Замер acceptance / TPF против AR-бейзлайна (lossless утверждается побитово)
./hf-auth.sh uv run python src/eval.py checkpoint=checkpoints/last.ckpt

# 5. Генерация обученным драфтером (greedy; --temperature для сэмплирования)
./hf-auth.sh uv run python main.py -p "..." \
    --checkpoint checkpoints/last.ckpt --jumps 2
```

Отладка на ноутбуке: `src/train.py` и `src/eval.py` работают и на маленьком
негейтированном бэкбоне — добавьте hydra-оверрайды
`model.name=HuggingFaceTB/SmolLM2-135M-Instruct model.backbone.dtype=float32
model.backbone.device_map=null`.

### Эксперименты (этапы задания)

Каждый этап задания проекта — один пресет, то есть готовый набор настроек (`src/configs/experiment/`). Чекпоинты падают
в `checkpoints/` (DF-голова + Adam-моменты, без бэкбона), кривые обучения — в TensorBoard:
`uv run tensorboard --logdir checkpoints`.

**Словарь кривых обучения:**

| Кривая | Что означает | Что ждать |
|---|---|---|
| `train/loss` | полный лосс шага | ↓ (шумно: каждый шаг — свежие `(s,t)` и точка разреза) |
| `loss/anchor` | `KL(sg(p_AR) ‖ π_{t,t})` — учится ли диагональ у верификатора | монотонно ↓ |
| `loss/ec` | согласие прыжка с диагональю в точке приземления | ↓ по мере обучения диагонали |
| `loss/td` | временной дрейф `‖∂_t π‖²` | всплеск после старта, затем умеренные значения; устойчивый 0 у обученной модели = мёртвый time-канал (сигнал к adaLN-апгрейду) |
| `loss/lambda` | текущий вес ECLD | рамп 0→λ при staged, константа иначе |
| `val/teacher_agreement` | доля позиций, где argmax драфтера = argmax верификатора; прокси acceptance | **главная кривая**: ↑ = проект едет |

**Этап 1 — бейзлайн Orthrus:**

```bash
./hf-auth.sh uv run python src/train.py +experiment=baseline
```

Одношаговая masked-diffusion голова самого Orthrus (барицентр как `[MASK]`)
на 0.5B бэкбоне, полноследовательный рецепт: случайная доля позиций
маскируется, голова восстанавливает их под один KL-член к замороженному
учителю. Смотреть: `train/loss` ↓, `val/teacher_agreement` ↑ (бейзлайн
стартует выше flow-вариантов — реконструкция при видимых соседях проще
транспорта из шума).

**Этапы 2–3 — flow-map драфтер + staged dual distillation:**

```bash
./hf-auth.sh uv run python src/train.py +experiment=flowmap_staged
```

CFM-драфтер в той же геометрии; λ рампится 0→1 за 2000 шагов — сначала
учитель (якорь), затем консистентность, как требует задание. Смотреть:
`loss/anchor` падает с первого шага; `loss/ec` включается по мере рампа
(виден на `loss/lambda`); `val/teacher_agreement` растёт медленнее
бейзлайна на старте — норма, драфт из чистого шума требует больше шагов.
Вариант со статичным λ (без staging):
`./hf-auth.sh uv run python src/train.py model=qwen2_0.5b`.

**Этап 4 — lossless при сэмплировании:**

```bash
# coupled (дефолт): побитовый lossless при T>0 — тот же сид даёт тот же текст, что AR
./hf-auth.sh uv run python src/eval.py model=qwen2_0.5b checkpoint=<ckpt> decode.temperature=0.8
# uncoupled (Левиафан): эквивалентность законов, нуль-калиброванный TV-тест
./hf-auth.sh uv run python src/eval.py model=qwen2_0.5b checkpoint=<ckpt> \
    decode.temperature=0.8 decode.coupled=false decode.equiv_samples=500
```

**Этап 5 (абляции) — вклад каждого члена дистилляции:**

```bash
./hf-auth.sh uv run python src/train.py +experiment=ablate_teacher_only        # только якорь (lambda=0)
./hf-auth.sh uv run python src/train.py +experiment=ablate_consistency_only    # только консистентность (anchor_weight=0)
```

Ожидания: teacher-only — диагональ учится, прыжки без сигнала → acceptance
низкий; consistency-only — самосогласованность без правды → acceptance
коллапсирует. Обе строки идут в таблицу абляций отчёта.

**Этап 5 (оценка) — валидация на датасете и таблица результатов:**

Датасет оценки — hydra-группа `configs/data/`. Датасетов два, и они отвечают на разные вопросы:

- `data=math500` (**дефолт eval**) — **другой датасет**, которого не было в
  обучении (HuggingFaceH4/MATH-500) — драфтер видит такие тексты впервые,
  поэтому это честная цифра обобщения. Задачи оборачиваются chat-шаблоном верификатора
  (user-ход + generation prompt) и декодируются с полного промпта
  (`decode.prompt_len=null`);
- `data=nemotron` — обучающее распределение: конкретные валидационные
  сэмплы в обучение не попадали (train их пропускает), но тексты того же
  сорта, что и обучающие.

В отчёт идут обе цифры — разрыв между ними показывает, насколько драфтер
переобучен под обучающую смесь. Метрики: **acceptance ± std** и **TPF ± std** (заголовочные),
tokens/s + speedup против AR (диагностика), NLL продолжения (сэмплирование).
Lossless утверждается **побитово** — падение прогона при расхождении.

```bash
# (i) AR-бейзлайн — знаменатели метрик (tpf_ar, tokens_per_s_ar), отдельный прогон не нужен
# (ii) Orthrus masked-diffusion
./hf-auth.sh uv run python src/eval.py model=qwen2_0.5b variant=baseline checkpoint=<ckpt>
# (iii) flow-map, 1 прыжок (основной замер)
./hf-auth.sh uv run python src/eval.py model=qwen2_0.5b variant=fixed checkpoint=<ckpt>
# (iv) flow-map, несколько прыжков: кривая acceptance от числа проходов (сетка K x jumps).
#      ВАЖНО: decode.block_size = K, число токенов черновика за цикл на ИНФЕРЕНСЕ —
#      ручка ЛЮБОГО варианта (включая fixed); с вариантом ОБУЧЕНИЯ block_wise она
#      не связана, несмотря на похожее имя
./hf-auth.sh uv run python src/eval.py -m model=qwen2_0.5b variant=fixed checkpoint=<ckpt> \
    decode.block_size=4,8,16 decode.jumps=1,2,4
# каждая строка выше — на MATH-500, которого не было в обучении (дефолт data=math500);
# парная цифра на обучающем распределении — тот же прогон с data=nemotron:
./hf-auth.sh uv run python src/eval.py model=qwen2_0.5b variant=fixed checkpoint=<ckpt> data=nemotron
```

**Дополнение (сверх задания) — обучение в инференсной геометрии:**

Этапы задания выше используют полноследовательные варианты (`fixed` /
`baseline`) — в самом задании про геометрию обучения ничего не сказано. Наше
дополнение: те же два драфтера, переобученные в точной геометрии
декодирования (block-causal: каузально к закэшированному префиксу,
двунаправленно внутри блока, чистый якорь) — варианты `block_wise` /
`baseline_block_wise`. Это убирает расхождение геометрий обучения и
инференса и даёт сравнение «геометрия-в-геометрию»:

```bash
./hf-auth.sh uv run python src/train.py +experiment=baseline_block_wise
./hf-auth.sh uv run python src/train.py +experiment=flowmap_block_wise
# оценка: те же команды этапа 5 с variant=baseline_block_wise / block_wise
```

**Формулы метрик и что чему соответствует:**

Цикл декодирования `c` принимает `a_c` драфт-токенов (0 ≤ a_c ≤ K) и всегда
добавляет 1 токен от верификатора (коррекцию или бонус):

```
tokens_per_cycle_c = a_c + 1
acceptance         = (1/C) * Σ_c a_c                  # средний приём за цикл
n_forwards         = 1 (префилл) + C * (jumps + 1)    # jumps драфт-проходов + 1 верификация
TPF                = N / n_forwards ≈ (acceptance + 1) / (jumps + 1)   # при N >> префилла
TPF_ar             = 1.0                              # ровно 1 токен на forward
теоретический speedup = TPF / TPF_ar                  # по проходам; wall-clock зависит от ядра
```

Соответствие между обучением и оценкой: `val/teacher_agreement` —
по-позиционный прокси приёма (доля argmax-совпадений драфтера с
верификатором); `val/acceptance_decode` и `val/tpf` — те же `acceptance` и
`TPF`, измеренные настоящей петлёй на валидационных промптах во время
обучения; `acceptance`/`tpf` в `src/eval.py` — они же на полноразмерной
оценке (`n_prompts` × `max_new_tokens`, ± std). Драфтер «бьёт бейзлайн»,
когда его TPF выше при том же числе прыжков — то есть когда
`acceptance_flow > acceptance_baseline` при jumps=1.

**Кривые целевых метрик и анализ результатов:**

Во время обучения, помимо лоссов, на каждой валидации гоняется настоящая
decode-петля (одношаговая) на `train.val_decode_prompts` промптах — в
TensorBoard появляются кривые **`val/tpf`** и **`val/acceptance_decode`**:
это ровно те метрики, по которым проект сравнивается с бейзлайном, видимые
по ходу обучения. Чекпоинт отбирается по лучшему `val/tpf`
(`train.monitor`); осторожно — на малом числе промптов метрика шумная,
поднимите `val_decode_prompts` до 8+, если селекция дёргается, либо
вернитесь на `monitor: val/loss`. Ранняя остановка при росте лосса —
`train.early_stop_patience` (валидаций подряд без улучшения val/loss).

Каждый прогон `src/eval.py` дописывает JSON-строку в `results/eval.jsonl`
(вариант, чекпоинт, K, jumps, температура, coupled, все метрики ± std).
Накопив прогоны по конфигурациям — бейзлайн/flow, сетка K × jumps, greedy и
сэмплирование — постройте фигуры отчёта одной командой:

```bash
uv run python src/plots.py           # results/eval.jsonl -> results/*.png
```

- `frontier.png` — кривая acceptance от числа проходов (линия на каждую пару
  вариант × K): главный график статьи — «сколько acceptance покупает
  каждый дополнительный прыжок»;
- `tpf.png` — TPF ± std по конфигурациям с горизонтальной AR-референс-линией:
  всё, что выше неё, реально ускоряет декодирование;
- `block_size.png` — TPF от размера блока K по вариантам.

### Контекст: узкое место декодирования

- **AR LLM** декодируют строго последовательно: *L* токенов → *L* прямых проходов (упор в пропускную способность памяти).
- **Диффузионные LM** драфтят блоки параллельно, но дрейфуют от AR-распределения и теряют качество.
- **Верификация в стиле спекулятивного декодирования** чинит качество: драфтим параллельно, затем *сверяемся* с AR-моделью → оставляем только корректные токены (**lossless**).

### Host-фреймворк: Orthrus

FlowDraft строится внутри **Orthrus** — каркаса lossless-параллельного декодирования:

- Один трансформер, две attention-ветки: **замороженная AR-ветка** и **лёгкая обучаемая диффузионная ветка** (~16% параметров), использующие общие norm / MLP / эмбеддинги и единый KV-кэш.
- Диффузионная ветка предлагает *K* токенов параллельно; замороженная AR-голова верифицирует их за один проход → вывод **доказуемо идентичен** базовой модели. Принятые токены коммитятся в общий KV-кэш, и цикл продолжается со следующего блока.
- По данным Orthrus: ускорение до **7.8×**, обучается лишь **~16%** параметров на **<1B** токенов.

> *Эти цифры относятся к host-фреймворку Orthrus (предыдущая работа), а не к результатам самого FlowDraft.*

### Проблема

- Пропускная способность любой системы с верификацией = **длина приёма** (принятых сдрафченных токенов за цикл).
- Драфтер Orthrus — **одношаговая masked diffusion** модель → предполагает, что позиции в блоке условно независимы → драфты расходятся → токены отклоняются.
- Уточнение драфта помогло бы, но **лишний шаг стоит прямого прохода** и снижает пропускную способность.
- Нужно **лучшее предложение за проход**, а не больше проходов.

### Идея: драфтер на Categorical Flow Map

- **Categorical Flow Maps** [Roos et al., 2026] выучивают *интегрированное, скоррелированное* распределение конечной точки на симплексе и генерируют за **один или несколько скачков (jumps)**.
- Используем это как драфтер: **более качественное совместное предложение** по блоку — при **том же числе проходов**.
- Верификация не меняется → вывод остаётся **строго lossless**; драфтер влияет только на *скорость*, но не на *качество*.
- **Новизна:** flow-map драфтер + **двойная дистилляция (dual distillation)**, подгоняющая одновременно распределение AR-учителя **и** консистентность flow-map.

**Почему это важно**

1. **Эффективность** — больше длина приёма = выше пропускная способность, бесплатно.
2. **Точность (fidelity)** — ускорение при **нулевой** потере качества (это гарантирует верификация).
3. **Фундамент** — связывает дистилляцию flow-map с быстрым и точным инференсом LLM.

### Коротко про dual distillation

Драфтер обучается с **двумя учителями** — отсюда *dual* — потому что один учитель не может дать оба нужных навыка:

- **От AR-модели — *что* предлагать.** Драфтер учится воспроизводить распределение, которое принял бы замороженный AR-верификатор, а не текст обучающего корпуса. Цель — быть *принятым верификатором*, а не совпасть с корпусом.
- **От самого себя — *как* прыгать.** Надёжный «локальный» режим драфтера учит его же более сложный «дальний прыжок», чтобы выдавать весь блок за один или несколько прыжков. Это и есть консистентность flow-map (self-consistency).

Оба учителя используются как фиксированные цели (stop-gradient) — градиент в них не течёт. Баланс между двумя членами — главный настроечный параметр.

Замороженная AR-модель на обучении используется **только как учитель**: она даёт целевое распределение, её веса не обновляются. На инференсе она тоже заморожена и работает как верификатор — тот же факт, что и гарантирует lossless.

### Цели

1. **Воспроизвести Orthrus** (замороженный AR + masked-diffusion драфтер, общий KV-кэш, lossless-петля) в приемлемом масштабе.
2. **Реализовать flow-map драфтер** (симплексная голова конечной точки, 1–несколько скачков).
3. **Разработать объектив двойной дистилляции** (распределение AR-учителя + консистентность flow-map).
4. **Оценить и сравнить:** AR-бейзлайн vs. masked-diffusion Orthrus vs. flow-map драфтер — по длине приёма, TPF и пропускной способности — всё с проверкой на lossless.

### Ожидаемые результаты

1. Воспроизведение lossless-параллельного декодера Orthrus (masked-diffusion драфтер).
2. Реализация **драфтера Categorical Flow Map** + обучение с двойной дистилляцией.
3. Оценка: сравнение по длине приёма / TPF / пропускной способности, с проверкой lossless и **абляциями по размеру блока / числу скачков**.

### Метод

Один замороженный бэкбон, два attention-пути (каркас Orthrus), CFM-драфтер на диффузионном пути и объектив двойной дистилляции. Реализовано; масштабная валидация впереди.

- **Адаптер** (`src/models/base/df_adapter.py`): каждый `q/k/v_proj` получает обучаемого двойника-копию замороженного AR-веса (~14% параметров 3B). Роутинг stateless (`torch.func.functional_call`, дерево модулей бэкбона не модифицируется); norm / MLP / `o_proj` / эмбеддинги / LM-head и единый KV-кэш — общие. Кэш AR-only по контракту: драфтер читает закоммиченный префикс, его собственные K/V срезаются сразу после каждого прохода. DF-путь работает **без маски** (двунаправленно; CFM не требует маски, кроме паддинга) и кондиционируется временами прыжка `(s, t)` через синусоидальный time-эмбеддинг с нулевой инициализацией (`fte.py`).
- **Объектив** (`FlowMapOrthrus.compute_loss`): `loss = anchor + λ · (4·EC + 2·TD)`
  - **anchor** — `KL(sg(p_AR) ‖ π_{t,t}(·))`: AR-верификатор заякоривает только диагональ (soft-таргет). Точка оценки — ручка `train.anchor_point`: trajectory `x_t` (дефолт) | landing `X_{s,t}(x_s)`. Прыжки прямого AR-лосса не получают — такой таргет константен по шумовому зерну и схлопывает транспорт.
  - **EC** — ур. (18) из *Categorical Flow Maps*: `CE(sg(π_{t,t}(X_{s,t}(x_s))), π_{s,t}(x_s))` — прыжки учатся у диагонали в точке собственного приземления; знание течёт `p_AR → π_{t,t} → π_{s,t}`.
  - **TD** — ур. (16): временной дрейф `‖∂_t π_{s,t}‖²`.
  - Пары `(s, t)` на сэмпл (`train.time_sampling`): `triangle` (равномерно на {s≤t}) | `sequential` | `paper` (t~U, s~U[0,t]).
- **Геометрии обучения** (`train.variant`): варианты из задания — полноследовательные: `fixed` (шумится вся последовательность) и `baseline` (одношаговый masked-diffusion драфтер самого Orthrus: без времени, барицентр как симплекс-нативный `[MASK]`). Наше **дополнение сверх задания**: `block_wise` / `baseline_block_wise` — те же два драфтера, переобученные в точности в инференсной геометрии (чистый AR-префикс в KV-кэше, ЧИСТАЯ якорная позиция блока — pending-токен decode-петли, см. ниже — и шумный K-блок; заодно сжимает тензоры лосса `[B,T,V]` → `[B,K,V]`).
- **Декодирование** (`FlowMapOrthrus.generate`): драфт K свежих токенов за 1–несколько прыжков → ОДИН AR-forward верифицирует блок. Коррекция/бонус прошлого цикла не коммитится отдельным проходом: она едет чистым якорем внутри блока, и следующая верификация коммитит её K/V, одновременно оценивая драфты — **цикл = `jumps + 1` forward'ов** (паритет TPF с конвенцией Orthrus). `temperature=0`: жадная верификация, выход **побитово** равен `ar_generate`. `temperature>0` с Gumbel-связыванием (дефолт): пер-позиционный Gumbel-шум делает сэмплирование детерминированным argmax'ом — выход **побитово** равен сэмплирующему `ar_generate` с тем же сидом. Без связывания (`coupled=false`): спекулятивное сэмплирование Левиафана, lossless **по распределению**.

### Структура репозитория

```text
FlowDraft/
├── main.py                        # playground-CLI (typer): генерация из ваших промптов
├── hf-auth.sh                     # HF_TOKEN из .env -> окружение (gated Llama)
├── pyproject.toml                 # uv-проект; ставится editable-пакетом `src`
└── src/
    ├── models/
    │   ├── base/df_adapter.py     # OrthrusAttentionAdapter: замороженный AR + DF-двойники
    │   ├── base/fte.py            # FlowTimeEmbedding (s, t)
    │   ├── model.py               # build_model: бэкбон + токенизатор + процессор
    │   ├── factory.py             # build_lit: выбор варианта + загрузка чекпоинта
    │   ├── lit_orthrus.py         # FlowMapOrthrus: лосс, обучение, lossless generate
    │   ├── lit_orthrus_block_wise.py      # ДОПОЛНЕНИЕ: обучение в инференсной геометрии
    │   ├── lit_orthrus_baseline.py        # masked-драфтер Orthrus (full-sequence)
    │   └── lit_orthrus_baseline_block_wise.py  # ДОПОЛНЕНИЕ: masked-драфтер, block-causal
    ├── preprocessor/df_processor.py   # токенизация + one-hot вершины симплекса
    ├── data/dataloaders.py        # стриминговый Dataset / collate / DataLoader;
    │                              #   EpochShuffled: повторы в новом порядке (эпохи)
    ├── configs/                   # hydra-конфиги
    │   ├── train.yaml             # конфиг точки входа обучения
    │   ├── eval.yaml              # конфиг точки входа оценки
    │   ├── model/                 # llama3_3b (дефолт) | qwen2_0.5b (0.5B из задания)
    │   ├── data/                  # nemotron (обучение) | math500 (оценка, не было в обучении)
    │   └── experiment/            # по пресету на этап задания + дополнения:
    │                              #   baseline | flowmap_staged | ablate_teacher_only |
    │                              #   ablate_consistency_only | baseline_block_wise |
    │                              #   flowmap_block_wise
    ├── train.py                   # точка входа обучения
    ├── eval.py                    # оценка на датасете: acceptance / TPF / NLL -> results/eval.jsonl
    └── plots.py                   # фигуры отчёта: frontier / TPF-бары / TPF-от-K
```

### Установка

```bash
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env     # доступ к gated meta-llama
./hf-auth.sh                      # проверить аутентификацию
```

### Использование

```bash
# генерация из ваших промптов (greedy: с побитовой lossless-проверкой)
./hf-auth.sh uv run python main.py -p "Once upon a time" -p "def main():"
# сэмплирование — тоже побитово равно AR (Gumbel-связывание — дефолт; --no-coupled = lossless по распределению)
./hf-auth.sh uv run python main.py -p "..." --temperature 0.8 --top-k 50 \
    --jumps 2 --checkpoint checkpoints/last.ckpt
```

### Обучение

Данные: [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) —
стриминг (без полной закачки), интерливинг категорийных сплитов, `messages`
рендерятся chat-template'ом токенизатора (`src/data/dataloaders.py`). Контракт
батча: `input_ids [B,T]` + `attention_mask [B,T]`; симплекс `[B,T,V]` строится
на девайсе и в батч не кладётся.

```bash
./hf-auth.sh uv run python src/train.py                            # fixed (рецепт задания)
./hf-auth.sh uv run python src/train.py train.variant=baseline     # бейзлайн Orthrus (из задания)
./hf-auth.sh uv run python src/train.py train.variant=block_wise   # ДОПОЛНЕНИЕ: инференсная геометрия
```

Ручки — в `configs/train.yaml`: `lambda`/`anchor_weight`/`lambda_ramp_steps`
(двойная дистилляция + staging), `anchor_point`, `time_sampling`,
`block_size`/`min_prefix` (block-wise варианты), `val_decode_prompts`
(decode-петля на валидации → кривые `val/tpf` + монитор чекпоинтов),
`early_stop_patience`, оптимизатор, Lightning `trainer.*`. Чекпоинты хранят
DF-голову + её Adam-моменты (~5 ГБ для 3B; замороженный бэкбон не пишется
никогда). Полный построчный справочник — ниже.

**Повторение потока (эпохи).** Датасет стриминговый, поэтому «эпоху»
определяете вы. Каждая новая эпоха Trainer открывает поток заново в НОВОМ
порядке (перемешивание пересеивается по эпохам; валидационный срез
отрезается до перемешивания и в обучение не утекает). Ограничить повторение
можно двумя способами:

```bash
# фиксированный пул сэмплов, повторённый N раз — «K примеров x N эпох» как в статье:
uv run python src/train.py trainer.max_steps=-1 trainer.max_epochs=2 data.train_size=471952
# или по числу шагов на повторение вместо числа сэмплов:
uv run python src/train.py trainer.max_steps=-1 trainer.max_epochs=3 trainer.limit_train_batches=2000
```

Без `data.train_size` каждое повторение берёт из огромного потока СВЕЖИЕ
сэмплы (больше разнообразия, но это не строгие эпохи) — с ним повторяется
ровно тот же пул в новом порядке.

**LR-расписание.** По умолчанию: линейный разогрев (5% шагов), затем
косинусный спад к нулю — пик задаёт `train.lr`, горизонт берётся из
`trainer.max_steps` (или `limit_train_batches` × `max_epochs`), текущее
значение видно кривой `lr-AdamW`. `train.lr_schedule=constant` отключает.

Всё вместе — учебный бюджет статьи Orthrus (2 эпохи по 472K упакованных
последовательностей ≈ 1.9B токенов, глобальный батч 128 × 2048 токенов,
cosine 2e-4 с 5% разогревом):

```bash
./hf-auth.sh uv run python src/train.py +experiment=baseline \
    data.train_size=471952 data.batch_size=8 data.max_length=2048 \
    trainer.accumulate_grad_batches=16 train.lr=2e-4 \
    trainer.max_epochs=2 trainer.max_steps=7375
```

(`max_steps` = 471952 сэмплов / (8 × 16) на оптимизаторный шаг × 2 эпохи;
косинусному расписанию он нужен явно — у стримингового загрузчика нет длины.)

### Справочник конфигов

Все конфиги — в `src/configs/` (hydra). Любой ключ переопределяется из
командной строки (`train.lr=3e-4`), группы меняются целиком
(`model=qwen2_0.5b data=nemotron`), пресеты добавляются `+experiment=...`.

**`train.yaml` — обучение (`src/train.py`)**

| Ключ | Дефолт | Что делает |
| --- | --- | --- |
| `seed` | 42 | глобальный сид: перемешивание данных, шум, инициализация |
| `output_dir` | `checkpoints` | куда падают чекпоинты и логи TensorBoard |
| `train.variant` | `fixed` | какой драфтер учить: из задания — `fixed` \| `baseline` (полноследовательные); дополнение — `block_wise` \| `baseline_block_wise` (инференсная геометрия) |
| `train.block_size` | 64 | K — длина блока на обучении (block-wise варианты) |
| `train.min_prefix` | 1 | минимальный чистый префикс перед блоком |
| `train.lr` / `weight_decay` / `betas` | 1e-4 / 0.01 / [0.9, 0.95] | AdamW только по DF-голове; `lr` — ПИК расписания |
| `train.lr_schedule` | `cosine` | `cosine` (линейный разогрев → косинусный спад к 0; нужен конечный `trainer.max_steps` или `limit_train_batches`+`max_epochs`) \| `constant` |
| `train.warmup_ratio` | 0.05 | только cosine: доля шагов на разогрев |
| `train.time_sampling` | `triangle` | как сэмплируются пары (s, t): `triangle` \| `sequential` \| `paper` |
| `train.lambda` | 1.0 | вес consistency-части (4·EC + 2·TD) |
| `train.anchor_weight` | 1.0 | вес AR-якоря; 0 = абляция «без учителя» |
| `train.lambda_ramp_steps` | 0 | staged-дистилляция: lambda 0 → `lambda` за N шагов; 0 = статично |
| `train.anchor_point` | `trajectory` | где якорь берёт диагональ: `trajectory` = π_{t,t}(x_t) \| `landing` = π_{t,t}(X_{s,t}(x_s)) |
| `train.checkpoint_name` | `flowdraft-{step:07d}` | шаблон имени чекпоинта — задавайте свой на каждый эксперимент (в CLI брать в кавычки: `'train.checkpoint_name="my-run-{step:07d}"'`) |
| `train.checkpoint_every_n_steps` | 1000 | как часто сохранять |
| `train.val_decode_prompts` / `val_decode_max_new` | 2 / 32 | настоящая петля декодирования на N val-промптах каждую валидацию → `val/tpf`, `val/acceptance_decode`; 0 = выкл |
| `train.monitor` / `monitor_mode` | `val/tpf` / `max` | по какой кривой выбирается лучший чекпоинт |
| `train.early_stop_patience` | 5 | стоп после N валидаций без улучшения `val/loss`; 0 = выкл |
| `trainer.*` | — | прокидывается в `lightning.Trainer` как есть (precision, max_steps, …) |

**`eval.yaml` — метрики на датасете (`src/eval.py`)**

| Ключ | Дефолт | Что делает |
| --- | --- | --- |
| `checkpoint` | null | обученная DF-голова `.ckpt`; null = необученный драфтер |
| `variant` | `fixed` | должен совпадать с тем, как учили чекпоинт |
| `results_file` | `results/eval.jsonl` | каждый прогон дописывает JSON-строку (вход `src/plots.py`) |
| `decode.block_size` / `decode.jumps` | 8 / 1 | K и число уточняющих проходов на инференсе — ручки ЛЮБОГО варианта; `block_size` НЕ связан с вариантом обучения `block_wise` (см. гид ниже) |
| `decode.max_new_tokens` | 64 | сколько токенов генерировать на промпт |
| `decode.n_prompts` | 64 | сколько промптов взять из датасета (100–200 для таблицы) |
| `decode.prompt_len` | null | null = полный отрендеренный промпт; число N = первые N токенов |
| `decode.temperature` / `top_k` / `top_p` | 0 / null / null | 0 = greedy; >0 = сэмплирование |
| `decode.coupled` | true | T>0: Gumbel-связанное сэмплирование — побитово равно AR |
| `decode.equiv_samples` | 0 | только uncoupled: N выборок для TV-теста равенства законов; 0 = выкл |

**`model/*` — бэкбон** (`llama3_3b` дефолт, `qwen2_0.5b` — масштаб 0.5B, который просит задание):
`name` (HF id), `backbone.dtype`, `backbone.device_map`,
`backbone.attn_implementation` (`sdpa` дефолт \| `flex_attention` только GPU \| `eager`).

**`data/*` — датасет** (`nemotron` для обучения, `math500` — не встречался в обучении — для оценки):
`dataset` (HF id), `splits`, `text_field` (колонка для текстовых бенчей),
`streaming`, `shuffle_buffer`, `val_size` (первые N сэмплов потока → валидация),
`train_size` (null = весь поток; число N = фиксированный пул из N сэмплов —
`trainer.max_epochs` повторяет именно их), `batch_size`, `max_length`, `num_workers`.

**`experiment/*` — по пресету на этап задания, плюс пресеты-дополнения.**
Каждый задаёт свои `output_dir` и `train.checkpoint_name` — прогоны не
перетирают друг друга (в том числе `last.ckpt`):

| Пресет | Задаёт | Чекпоинты |
| --- | --- | --- |
| `baseline` | `variant=baseline` | `checkpoints/baseline/baseline-*.ckpt` |
| `flowmap_staged` | `variant=fixed`, `lambda_ramp_steps=2000` | `checkpoints/flowmap-staged/flowmap-staged-*.ckpt` |
| `ablate_teacher_only` | `variant=fixed`, `lambda=0` | `checkpoints/ablate-teacher/ablate-teacher-*.ckpt` |
| `ablate_consistency_only` | `variant=fixed`, `anchor_weight=0` | `checkpoints/ablate-consistency/ablate-consistency-*.ckpt` |
| `baseline_block_wise` (дополнение) | `variant=baseline_block_wise` | `checkpoints/baseline-block-wise/baseline-block-wise-*.ckpt` |
| `flowmap_block_wise` (дополнение) | `variant=block_wise`, `lambda_ramp_steps=2000` | `checkpoints/flowmap-block-wise/flowmap-block-wise-*.ckpt` |

Свой эксперимент (например, серия по `anchor_point` после основных этапов,
или block-wise против fixed) — переопределите имя и каталог, чтобы у него
тоже были свои каталог и имя:

```bash
./hf-auth.sh uv run python src/train.py +experiment=flowmap_staged \
    train.anchor_point=landing \
    output_dir=checkpoints/anchor-landing 'train.checkpoint_name="anchor-landing-{step:07d}"'
```

### Параметры инференса простыми словами

Один цикл декодирования устроен так: драфтер угадывает сразу целый блок
токенов, замороженная базовая модель проверяет догадку за один проход,
совпавшее начало блока принимается, и базовая модель добавляет один свой
токен (исправление первой ошибки — или бонус, если совпало всё). Дальше —
следующий цикл. Ручки:

- `--block-size` (K) — сколько токенов драфтер угадывает за цикл. Больше —
  выше потенциальное ускорение, но хвост длинной догадки опирается на её же
  непроверенное начало и отбрасывается чаще. Перебирайте 4–16.
  Несмотря на похожее имя, к варианту обучения `block_wise` это отношения
  не имеет — блоками на инференсе работает любой драфтер.
- `--jumps` — сколько проходов драфтер тратит на «полировку» догадки, прежде
  чем показать её базовой модели. Каждый лишний проход улучшает догадку, но
  и стоит один forward: цикл обходится в `jumps + 1` проходов. Больше прыжков
  окупается, только если дополнительно принятые токены перевешивают
  дополнительные проходы.
- `--max-new-tokens` — потолок длины ответа.
- `--temperature` — 0: всегда самый вероятный токен; выход гарантированно
  совпадает с чистой базовой моделью, это проверяется побайтово. Больше 0 —
  случайное сэмплирование, текст живее.
- `--top-k` / `--top-p` — только при сэмплировании: ограничить выбор k самыми
  вероятными токенами / наименьшим набором с суммарной вероятностью p.
- `--coupled` (включено по умолчанию) — при сэмплировании драфтер и базовая
  модель берут случайность из одного общего источника с сидом. В итоге даже
  *сэмплированный* текст — ровно тот, что выдала бы чистая базовая модель с
  этим сидом, токен в токен. `--sampling-seed` выбирает, какой именно это
  текст; `--no-coupled` — классическое спекулятивное сэмплирование (то же
  распределение, но не те же токены).
- `--variant` + `--checkpoint` — геометрия драфтера и его обученные веса.
  Без чекпоинта драфтер необучен: выход всё равно точный, просто почти ничего
  не принимается (медленно).
- `--model` — бэкбон: имя конфига (`qwen2_0.5b`) или HF id.

Ни одна из этих ручек не меняет, *что* будет сгенерировано (сверх гарантий
выше) — только как быстро. Последнее слово по каждому токену — за
верификатором.

### Оценка

Промпты датасета (по умолчанию полные; `decode.prompt_len=N` — первые N
токенов) декодируются дважды — flow-draft и чистый AR — и сравниваются. Жадный lossless утверждается **побитово**, а не предполагается.

```bash
./hf-auth.sh uv run python src/eval.py checkpoint=path.ckpt   # variant=fixed по умолчанию
# сетка абляций размер-блока / число-прыжков (hydra multirun):
./hf-auth.sh uv run python src/eval.py -m decode.block_size=4,8,16 decode.jumps=1,2,4
```

Головные метрики (mean ± std по `n_prompts`): **acceptance** за цикл и **TPF**
(токенов на forward; цикл = `jumps+1`). Wall-clock tokens/s и speedup против
AR — диагностика (зависит от железа/ядра). Ядро внимания — переключатель
конфига (`model.backbone.attn_implementation`): `sdpa` (дефолт; fused,
поддерживает DF-маску — сверено с eager) | `flex_attention` (компилируемые
блок-маски, только GPU) | `eager` (референс). **NLL продолжения** под
замороженным учителем считается только в сэмплирующем режиме (при greedy
выход побитово равен AR и NLL не измеряет ничего). Дефолтный датасет
оценки — **MATH-500** (`data=math500` — этих текстов не было в обучении);
`data=nemotron` — оценка на обучающем распределении (валидационные сэмплы
в обучение не попадали, но тексты того же сорта). Разрыв между двумя цифрами —
переобученность драфтера под обучающую смесь. Сэмплирующая оценка
(`decode.temperature>0`): с Gumbel-связыванием (дефолт) выход по-прежнему
побитово равен AR; только `decode.coupled=false` даёт lossless по
распределению — там вместо побитового флага работает TV-тест
(`decode.equiv_samples`).

### Результаты

> 🚧 **TODO.** Заполнить после экспериментов. Все строки должны быть проверены на lossless.

| Метод | Длина приёма ↑ | TPF * | Пропускная способность (tok/s) ↑ | Lossless |
| --- | --- | --- | --- | --- |
| AR-бейзлайн | — | — | — | ✅ (тривиально) |
| Orthrus (masked-diffusion драфтер) | TBD | TBD | TBD | ✅ |
| **FlowDraft** (flow-map драфтер) | TBD | TBD | TBD | ✅ |

\* *TPF — токенов на прямой проход: `N сгенерировано / N проходов`, цикл = `jumps + 1` проходов (формулы — в гиде выше).*

**Абляции (TODO):** размер блока, число скачков.

### Ссылки

- **Categorical Flow Maps** — Roos et al., ICML 2026. arXiv:2602.12233. Референсная имплементация: `olsdavis/semicat`. <!-- TODO: сверить финальную ссылку и линки -->
- **Orthrus** — lossless-параллельное декодирование через замороженный AR-бэкбон + обучаемый диффузионный драфтер. arXiv:2605.12825. Референсная имплементация: `chiennv2000/orthrus`. <!-- TODO: сверить финальную ссылку и линки -->

<!-- TODO: дополнить, когда появятся метаданные -->
```bibtex
@misc{flowdraft2026,
  title  = {FlowDraft: Flow-Map Drafting for Lossless Parallel Decoding},
  author = {TODO},
  year   = {2026},
  note   = {Summer of Machine Learning at Skoltech (SMILES), Applied AI Center}
}
```

### Команда

- **Участники:** <!-- TODO: члены команды -->
- **Кураторы / менторы:** Maria Ivanova (YSDA, Applied AI Institute) · Dmitrii Babaev

### Благодарности

Работа выполнена в рамках **Summer of Machine Learning at Skoltech (SMILES)**, Skoltech Applied AI Center.

### Лицензия

> 🚧 **TODO:** выбрать и добавить лицензию (например, MIT / Apache-2.0).