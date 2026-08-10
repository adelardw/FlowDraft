# FlowDraft: Flow-Map Drafting for Lossless Parallel Decoding

> Raising the **acceptance ceiling** of lossless parallel decoding by upgrading the *drafter* to a **Categorical Flow Map** — faster generation, provably identical output.

**Documentation:** **English** · [Russian](README.ru.md)

<!-- Badges — TODO: fill in once the repo is public
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-WIP-orange)
-->

> 🚧 **Status: core implementation landed and smoke-verified** — adapter, three training variants (`flowdraft`, `flowdraft_block_wise`, and `orthrus`; trained end-to-end on real data: SmolLM2-135M + Nemotron), lossless decoding (**bitwise** at greedy AND at sampling via Gumbel coupling; `jumps+1` forwards per cycle), evaluation harness (mean±std, JSONL results + report plots), experiment presets for every stage of the task. GPU experiments pending; **Results** are TBD.

**Summer of Machine Learning at Skoltech (SMILES) · Applied AI Center**

---


## Table of contents

- [Multi-step drafting study (August 2026)](#multi-step-drafting-study-august-2026)
- [Overview](#overview)
- [Quickstart](#quickstart) — setup, sparse attention, training, validation, statistics, curves
- [Experiments (task stages)](#experiments-task-stages)
- [SmolLM2-135M bench: what the loss terms actually buy](#smollm2-135m-bench-what-the-loss-terms-actually-buy)
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

## Multi-step drafting study (August 2026)

Ten training runs on SmolLM2-135M at 20k steps, three of them replicated across
three seeds. Measured on 460 tasks from six benchmarks: 56,000 per-prompt
observations at one seed, plus 144 further measurements across seeds at the
common horizon. Full write-up, objective and mathematics:
[EXPERIMENTS.md](EXPERIMENTS.md).

**Headline.** Orthrus concludes that single-step projection is optimal. That
holds for a masked drafter and fails for a continuous state. Intervals below use
the **training seed** as the unit of observation (three seeds, df = 2), so they
describe the method rather than one trained model.

| contrast (three refinement passes, 20k steps) | Δ accepted tokens | 95% CI | p |
|---|---|---|---|
| multi-step training, continuous state | **+1.138** | ± 0.109 | 0.0005 |
| best run vs reproduced Orthrus | **+0.835** | ± 0.085 | 0.0006 |
| continuous state vs masking, same objective | +0.612 | ± 0.065 | 0.0006 |
| multi-step training, masking | +0.223 | ± 0.021 | 0.0005 |

Going from one refinement pass to four, acceptance grows by **+0.893** with a
continuous state trained on the procedure, by +0.070 with masking, and **falls
by 0.563** for the same continuous architecture left untrained on it — on all
three seeds.

**Multi-step buys quality, not speed.** Throughput falls monotonically and
nothing multi-step exceeds 1.0 tokens per forward, i.e. none beats plain
autoregressive decoding: a cycle costs `n+1` forwards while acceptance grows
slower than `n`. This is what the prefix-fixing lemma predicts — it yields
`TPF = 1` as a floor, not as a speedup mechanism. Only single-pass decoding
clears 1.0 (1.326 / 1.257 / 1.219).

**A reversal worth noting.** At a single pass masking *wins* (−0.148 ± 0.047).
The advantage of a continuous state appears only with refinement and grows with
it: −0.148 → +0.612 → +0.675 at one, three and four passes.

![acceptance and throughput vs refinement passes](results/figures/multistep.png)
![paired contrasts with the seed as the unit](results/figures/contrasts.png)
![training curves across three seeds](results/figures/seeds.png)
![per-benchmark breakdown](results/figures/per_benchmark.png)
![measured horizon](results/figures/horizon.png)

Training curves for every logged loss term and every logged metric, across all
three seeds, are in [step 6 of the Quickstart](#6-training-curves-and-figures):
colour is the experiment, line style is the seed, so how tightly the three
styles of one colour overlap *is* the between-seed spread.

Untrained multi-step refinement is not merely worse but *unpredictably* worse:
at four passes the three seeds give 1.227 / 0.933 / 0.778 (σ = 0.228), while
every other run stays within 0.05.

### Running the campaign on Qwen3-1.7B

The four experiments that carry the claims have Qwen presets reproducing the
paper's Table 4 hyperparameters exactly (2048 tokens, 256 anchor blocks, block
size 32, two epochs over 600k examples, peak LR 2e-4 cosine with 5% warmup,
gradient clipping 1.0, global batch 128, 1:1:1 chat/math/code).

```bash
# reference point: Orthrus verbatim — W_Q, W_K, W_V only
./hf-auth.sh uv run python src/train.py +experiment=qwen_masked_paper

# masked drafter trained on its own refinement procedure
./hf-auth.sh uv run python src/train.py +experiment=qwen_masked_multistep

# continuous state, verifier alignment only — the ablation
./hf-auth.sh uv run python src/train.py +experiment=qwen_flow_baseline

# continuous state trained on its own refinement procedure — the main result
./hf-auth.sh uv run python src/train.py +experiment=qwen_flow_multistep
```

Measure a checkpoint. `model.backbone.dtype=float32` is required for the
losslessness assertion — under bf16 the verifier's arithmetic breaks bitwise
agreement on near-ties. `decode.jumps` takes restart pairs; an integer expands
to passes at `t<1` that nothing in the objective trains when the consistency
terms are off.

```bash
./hf-auth.sh uv run python src/eval.py \
    checkpoint=checkpoints/qwen-flow-selfcorrect/last.ckpt \
    data=math500 decode.block_size=32 decode.n_prompts=100 \
    "decode.jumps=[[0,1],[0.5,1],[0.75,1]]" \
    model.backbone.dtype=float32 \
    per_prompt_file=results/qwen-per-prompt.jsonl

uv run python bench/analyze.py --data results   # CIs, Holm, RM-ANOVA, bootstrap
```

**Memory, single device, micro-batch 1:** ~12.8 GB for the paper projection set
and ~14.5 GB for ours, plus attention activations — comfortable on an 80 GB
A100, worth measuring on a 40 GB card. The paper used 8 GPUs for throughput,
not because a single device cannot hold it: global batch 128 is micro-batch 1
with accumulation 128 on one device or 16 on eight, and the per-device
footprint is identical.

**One knob does not transfer.** `acceptance_profile` states the acceptance
regime the position weights aim at. It is 0.8 on the 135M bench and 0.93 in the
Qwen presets, interpolated from the paper's own numbers (TPF 6.35 at acceptance
length 11.7 solves to a = 0.929). Getting it wrong is not free: at 0.93 the
deep positions carry ~44% of the gradient mass, at 0.6 about 3%. Every run logs
per-position acceptance, so derive it from the data and retrain if it moved.

**Untested on this hardware.** Collective operations, the rank seed offset and
the sparse FlexAttention path are no-ops on a single MPS device and have never
been exercised. Run one short job before committing to a long one.

### What changed in the code

Defects found and fixed, each with a measured before/after:

| | before | after |
|---|---|---|
| commit widths in multi-step training | `[16, 31]` — the second pass supervised a block with no masks left | `[10, 21]`, matching decode exactly |
| validation schedule given as an integer | two passes of three ran at `t<1`, where no term provides gradient | restart pairs matching what is trained |
| position weights across the two branches | different dtype, 2.828 vs 2.824 | bitwise identical |
| mixed-benchmark validation | crashed on nested Hydra initialisation | works |
| schedule parsing for pair form | crashed on `ListConfig` | all five input forms |
| dead forward when consistency terms are off | 3 and 7 backbone forwards per step | 2 and 6; step time 0.18 s → 0.11 s |
| time conditioning consumed the global RNG | one architecture saw different data at the same seed | reseed after model construction |
| `eval.py` | measured a block-32 model at block 8; crashed writing paired schedules | fixed |
| `min_jump_gap` | justification was wrong: the gradient at `s≈t` is `O(t−s)`, not zero | left at 0 |

Portability: the CUDA path no longer refuses non-Qwen3 backbones; the
finite-loss check and the crash-checkpoint decision are collective; the seed is
offset by rank; all three teacher modes are chunked (6.09 GB → 0.75 GB at the
paper preset); `val_check_interval` in the paper presets counted loader batches
— 3.9 optimizer steps instead of 250.

Losslessness needs `model.backbone.dtype=float32`: under bf16 the verifier's
arithmetic breaks bitwise agreement on near-ties (5 of 6 vs 6 of 6).

Rejected configurations live in [bucket/](bucket/) with their numbers.
Objective assumptions that code cannot remove are in
section 6 of [EXPERIMENTS.md](EXPERIMENTS.md), including one measured and found unsatisfied.

## Overview

Autoregressive (AR) LLMs decode strictly sequentially: generating *L* tokens costs *L* forward passes, which is memory-bandwidth bound. Diffusion LMs can draft whole blocks in parallel, but they drift from the AR distribution and lose quality. Speculative-style verification restores quality: draft a block in parallel, then verify it against the AR model in a single pass and keep only the tokens the AR model would have produced — this is **lossless**.

**FlowDraft** upgrades the *drafter* inside a lossless parallel-decoding loop. The throughput of any verify-based system is governed by its **acceptance length** — the number of drafted tokens accepted per cycle. We replace the single-step masked-diffusion drafter with a **Categorical Flow Map** drafter that produces a higher-fidelity *joint* proposal over the block at the **same** number of forward passes. Verification is left untouched, so the output stays strictly lossless — the drafter affects only **speed**, never **quality**.

Crucially, the AR model is what does the verifying, so it is kept **frozen throughout**. Keeping it untouched is exactly what makes the output provably identical to the base model; it is what the word *lossless* rests on.

## Quickstart

The full campaign, in the order it has to run. Every step is resumable on its
own: training restarts from `last.ckpt`, and measurement skips any combination
already present in its output file, so an interrupted sweep is re-launched with
the same command.

### 1. Setup (once)

```bash
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env          # gated backbone access
./hf-auth.sh                           # verify: prints your HF username
```

Check inference before training anything — the **untrained** drafter is already
lossless, just slow:

```bash
./hf-auth.sh uv run python main.py -p "Once upon a time"
#   -> generation + [lossless vs greedy AR: PASS]
```

### 2. Sparse attention: FlexAttention and FlashAttention-4

The masked baseline drafts up to 256 isolated blocks per step, so its attention
mask is sparse by construction. Three backends implement the *same* mask —
causal into the cached prefix, bidirectional within a block, nothing across
blocks — and differ only in speed:

| backend | override | where it runs |
|---|---|---|
| FlexAttention + Triton | `model.adapter.flex_attention_backend=triton` | every CUDA architecture; the stable default |
| FlashAttention-4 | `model.adapter.flex_attention_backend=flash` | Hopper / Blackwell only (compute capability >= 9) |
| dense additive mask | chosen automatically | CPU, Apple Silicon, CUDA builds without FlexAttention |

FA4 ships as a prerelease and is deliberately **not** a project dependency, so
CPU and macOS environments stay lightweight. On the CUDA node:

```bash
uv pip install ninja packaging
uv pip install --prerelease=allow --no-build-isolation "flash-attn-4[cu13]"

# verify BOTH before committing to a long run
uv run python -c "from torch.nn.attention.flex_attention import flex_attention; print('FlexAttention: OK')"
uv run python -c "import flash_attn; print('flash-attn: OK')"
```

Then append `model.adapter.flex_attention_backend=flash` to the training
commands below. Asking for `flash` on a pre-Hopper card is refused with a
message rather than silently downgraded.

### 3. Train every experiment

Qwen3-1.7B — the four experiments that carry the claims, at the paper's
hyperparameters:

```bash
for exp in qwen_masked_paper qwen_masked_multistep \
           qwen_flow_baseline qwen_flow_multistep; do
  ./hf-auth.sh uv run python src/train.py +experiment=$exp \
      output_dir=checkpoints/$exp \
      model.adapter.flex_attention_backend=flash
done
```

SmolLM2-135M — the full bench, replicated across seeds. Runs go **one at a
time**: on a single device parallel runs contend for the same memory and the
timings stop being comparable.

```bash
EXPERIMENTS="smollm_masked_paper smollm_masked_baseline smollm_masked_selfcorrect \
      smollm_masked_selfcorrect_prefixweight smollm_flow_verify \
      smollm_flow_selfcorrect smollm_flow_selfcorrect_prefixweight"

for seed in 42 43 44; do
  out=checkpoints; [ $seed = 42 ] || out=checkpoints/s$seed
  for exp in $EXPERIMENTS; do
    resume=""
    [ -f "$out/$exp/last.ckpt" ] && resume="resume_from_checkpoint=$out/$exp/last.ckpt"
    ./hf-auth.sh uv run python src/train.py +experiment=$exp \
        seed=$seed output_dir=$out/$exp $resume
  done
done
```

Watch `val/acceptance_decode` rise and `val/loss/verify_kl` fall. Checkpoints
hold the FP32 drafter head and its Adam moments; the frozen backbone is not
stored. For multi-GPU, hand the run to Lightning's DDP:

```bash
./hf-auth.sh uv run python src/train.py +experiment=qwen_flow_multistep \
    trainer.accelerator=gpu trainer.devices=8 trainer.strategy=ddp
```

Training always disables `model.backbone.device_map` — Hugging Face device maps
are inference sharding, while DDP needs one complete replica per GPU. Streaming
train and validation datasets are split into disjoint, equal rank shards, then
partitioned among DataLoader workers. `data.batch_size` is per GPU; reach a
larger global batch with `trainer.accumulate_grad_batches`.

### 4. Validate on every dataset

Six benchmarks, 460 tasks. `model.backbone.dtype=float32` is **required** for
the losslessness assertion — under bf16 the verifier's arithmetic breaks bitwise
agreement on near-ties. `per_prompt_file` is what makes the statistics possible:
without it only means are written, and no interval, contrast or ANOVA can be
built from means alone.

```bash
CKPT=checkpoints/qwen_flow_multistep/last.ckpt

for ds in gsm8k:100 math500:100 humaneval:100 mbpp:100 aime24:30 aime25:30; do
  ./hf-auth.sh uv run python src/eval.py \
      checkpoint=$CKPT data=${ds%%:*} decode.n_prompts=${ds##*:} \
      decode.block_size=32 decode.max_new_tokens=64 \
      "decode.jumps=[[0,1],[0.5,1],[0.75,1]]" \
      model.backbone.dtype=float32 \
      results_file=results/aggregate.jsonl \
      per_prompt_file=results/pp-qwen.jsonl
done
```

AIME 24 and 25 hold 30 tasks each — that is the whole set, not a subsample.
`decode.jumps` takes restart pairs; an integer expands to passes at `t < 1`, which
nothing in the objective trains once the consistency terms are off. Sweep the
schedule by repeating the loop with `decode.jumps=1`, `[[0,1],[0.5,1]]` and the
four-pass form.

### 5. Metrics and statistics

```bash
uv run python bench/analyze.py --data results
```

Prints, in order: acceptance per experiment with Student intervals; every paired
contrast with Holm correction; repeated-measures ANOVA with the
Greenhouse-Geisser correction; the per-dataset breakdown; stratified bootstrap,
sign test and Wilcoxon across datasets; and rank stability across prompt folds
by Kendall tau. The unit of observation is the **prompt**, and the header states
plainly how many training seeds stand behind the numbers — resampling prompts
answers "will this hold on other tasks", never "will this hold on another
training run".

Useful flags: `--step` (which snapshot), `--sched` (`n1`/`n2`/`n3`/`n4`),
`--metric acceptance|tpf`, `--folds`.

### 6. Training curves and figures

```bash
uv run python bench/curves.py     # every loss term and every metric, per seed
uv run python bench/figures.py    # acceptance vs passes, horizon, contrasts
```

`bench/curves.py` reads the TensorBoard scalars each run writes under
`checkpoints/**/lightning_logs/` and produces two figures in which colour is the
experiment and line style is the training seed, so "do the experiments differ"
and "do the seeds agree" stay separate questions. Terms carrying weight zero are
named in the caption instead of being drawn as a flat line on an invented axis.

![loss curves per experiment and seed](results/figures/curves_loss.png)
![validation metrics per experiment and seed](results/figures/curves_metrics.png)

Live monitoring during a run: `uv run tensorboard --logdir checkpoints`.

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
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_packed_blockwise

# Stage 4 — lossless at sampling: coupled = bitwise; uncoupled = null-calibrated TV test
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b checkpoint=<ckpt> decode.temperature=0.8
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b checkpoint=<ckpt> decode.temperature=0.8 \
    decode.coupled=false decode.equiv_samples=500

# Stage 5 (ablations) — the contribution of each distillation term

# Stage 5 (evaluation) — default: MATH-500, a dataset never seen in training
#           (data=nemotron evaluates on the training distribution);
#           block-size x jumps grid -> results/eval.jsonl -> report figures.
#           NOTE: decode.block_size = total width K: one clean anchor + K-1
#           drafted tokens per cycle — a knob of EVERY variant; it is unrelated to the
#           flowdraft_block_wise TRAINING-geometry variant despite the similar name
./hf-auth.sh uv run python src/eval.py -m model=qwen3_1.7b variant=flowdraft checkpoint=<ckpt> \
    decode.block_size=4,8,16 decode.jumps=1,2,4
uv run python src/plots.py

# ADDITION (beyond the task) — FlowDraft retrained in the exact inference
# geometry (block-causal), directly comparable with Orthrus
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_block_wise
#   eval: same stage-5 commands with variant=orthrus / flowdraft_block_wise
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

## SmolLM2-135M bench: what the loss terms actually buy

A small stand for settling design questions before they cost a Qwen3 run.
SmolLM2-135M, nemotron, `block_size=8`, `anchors_per_sequence=1`, decoded on
**math500** (never seen in training). **Qwen3-1.7B is the next stage; nothing
here is a headline result.**

Unit of observation is the pair (prompt, seed): 24 prompts x 5 seeds =
**120 cells** per point, `decode.fixed_prior=true`. Both noise axes are real —
the same checkpoint on the same prompts drifts by up to 0.16 tokens across prior
draws — so every contrast is **paired on both**, never a difference of means.
Each configuration is measured at **three training horizons**, because a single horizon
cannot tell a converging gap from a noisy one, and because between-snapshot
spread on an unchanged objective reaches 0.5 tokens.

### The objective, term by term

```
L = w_verify · KL( p_AR ‖ pi_{0,1}(x0) ) · u_j                                [1]
  + w_self   · (1/r) SUM_k KL( p_AR(· | argmax q_{k-1}) ‖ pi_{s_k,1}(x_k) ) · v_j · u_j
  + w_end    · CE( x1, pi_{t,t}(x_t) )                                        [2]
  + lambda   · ( 4 · CE( sg pi_{t,t}(X_{s,t}(x_s)), pi_{s,t}(x_s) ) + 2 · ||d_t pi_{s,t}||^2 · gamma^2 )

x0 ~ prior,  x_s = (1-s) x0 + s x1,  gamma = (t-s)/(1-s),  X_{s,t}(x) = x + gamma (pi - x)

q_0  = pi_{0,1}(x0)                     the jump the decode loop executes
x_k  = (1-s_k) x0' + s_k q_{k-1}         restart carrying the previous pass's draft
q_k  = pi_{s_k,1}(x_k),  detached between rounds,  s_1 < ... < s_r in (s_min, 1)

u_j = chain-validity gate x prefix-survival weights dE/da_j
v_j = 1 on the verifier-confirmed prefix, `selfcorrect_tail_weight` past the break
```

[1] is the only pair the decode loop runs at `jumps=1`. Its target is
corpus-conditioned, so the drafter's state cannot change the target's *value*,
and reading that state cannot lower the loss **at any capacity**. [2] and the
`lambda` block are the categorical-flow-map structure.

**Self-correction is the only term whose target the state enters.** One frozen AR
forward over the drafter's own proposal yields `p_AR(· | ctx, draft_{<j})` plus
the exact greedy verdict; the map is trained to answer, from a state carrying
that draft, what the verifier would say about it. Writing one Jacobi sweep as
`T(ctx,d)_j = argmax p_AR(· | ctx, d_{<j})`: if the map realises `T`, then `n`
composed jumps fix positions `1..n`, so `A_n >= min(n, K-1)` — deterministic
verifier, no new information. That guarantee alone yields `TPF <= 1`, exactly AR
speed: the term makes multi-step **meaningful**, not cheap.

The masked baseline gets the same term in the only state masked diffusion can
express — some positions committed, the rest re-masked
(`Orthrus._masked_selfcorrect`) — which is precisely its own
confidence-ordered-unmasking schedule. Running the two against each other
separates *training for the procedure* from *the state being continuous*.

### The five configurations

Named by what their objective contains, not by their script key. All five share
the data, the schedule, the backbone and the budget; only the terms differ.

| name | terms switched on | script key |
|---|---|---|
| **masked** | Orthrus's masked-block KL — one component | `orthrus` |
| **masked + self-corr** | + `[2]` on the masked state (its own re-masking schedule) | `orthrus_ms` |
| **flow** | `[1]` only: verify KL on the simplex prior, position weights | `fd_base` |
| **flow + self-corr** | `[1]` + `[2]`: the jump schedule with the on-policy sweep | `fd_ms` |
| **flow + self-corr + CFM** | + `[3]` and `[4]`: diagonal anchor, EC, drift | `fd_cfm_ms` |

### Results

Two independent training seeds, six configurations, 6000 steps each, snapshots along the
way. Unit of observation is the **prompt** (24 of them): the masked drafter's
decode is deterministic — the prior does not enter it — so its five decode seeds
are bit-identical copies, and treating 120 cells as independent understated the
standard error by sqrt(5).

| configuration | A@1 (s42/s43) | A@2 | A@4 | growth A@4 − A@1 |
|---|---|---|---|---|
| `masked` | 1.391 / 1.454 | 1.803 / 1.707 | 2.595 / 3.140 | +1.204 (t=+4.15) / +1.686 (t=+5.24) |
| `masked + self-corr` | 1.301 / 1.358 | 1.425 / 1.577 | 2.728 / 2.517 | +1.427 (t=+4.34) / +1.159 (t=+5.02) |
| `flow` | 1.134 / 1.259 | 1.148 / 1.224 | 0.963 / 1.042 | -0.171 (t=-3.51) / -0.217 (t=-2.08) |
| `flow + self-corr` | 1.310 / 1.339 | 1.826 / 1.779 | 2.218 / 2.136 | +0.909 (t=+4.02) / +0.797 (t=+4.11) |
| `flow + self-corr + CFM` | 1.163 / 1.401 | 1.748 / 1.782 | 2.006 / 2.085 | +0.843 (t=+3.75) / +0.684 (t=+3.50) |
| `flow + self-corr, old weighting` | 1.383 / 1.436 | 1.708 / 1.698 | 2.055 / 2.059 | +0.672 (t=+3.35) / +0.622 (t=+3.92) |

| configuration | TPF@1 | TPF@2 | TPF@4 |
|---|---|---|---|
| `masked` | 1.211 | 0.918 | 0.773 |
| `masked + self-corr` | 1.165 | 0.834 | 0.724 |
| `flow` | 1.098 | 0.729 | 0.401 |
| `flow + self-corr` | 1.162 | 0.934 | 0.635 |
| `flow + self-corr + CFM` | 1.141 | 0.922 | 0.609 |
| `flow + self-corr, old weighting` | 1.205 | 0.901 | 0.611 |

### What survived

**One finding, and it survived everything: clustering, a leak fix, weight
parity between branches, and a second seed.**

Acceptance grows with the number of jumps only when the drafter is trained
against the verifier's verdict on its own draft. Without that term the flow-map
configuration *degrades* with jumps — `+1.255` and `+1.093` at `n=4`
(t = +5.09, +4.11) separate the two, and the configuration without
it moves the wrong way on both seeds. At `n=2` the same contrast is
`+0.678` / `+0.554` (t = +4.27, +4.03).

The verdict enters the objective three ways, all from one frozen AR forward
over the drafter's own proposal: as the **target** `p_AR(· | ctx, draft_{<j})`,
as the **state** the next round reads (confirmed positions settled, the
corrected one clean, the tail carrying its rejected guess), and as the
**weighting** across positions.

### What did not

- **Training the masked drafter for multi-step.** `-0.090` and
  `-0.096` at `n=1` — negative on both seeds once the masked branch
  also got the position weights and the leak in its self-correction forward was
  removed. An earlier `+0.105` at t = 3.61 was two artefacts stacked: an
  understated SE, and position weights on one side of the comparison only.
- **The continuous state over the masked one.** `+0.401` /
  `+0.202` at `n=2` — same sign, twice the spread, significant on
  neither seed.
- **The CFM structure.** `-0.212` / `-0.050` at `n=4`. Effect
  not detected; the earlier `-0.63` at t = -6.5 did not reproduce once the
  weight confound and the LR re-warmup were gone.
- **Multi-step as throughput.** TPF falls with `n` for every configuration. Only `n=1`
  beats plain AR. Multi-step buys acceptance, never speed.

And the masked baseline grows with jumps **without being trained for it at
all** — confidence-ordered unmasking already does this. Its growth
(2.595 / 3.140) is at least as large as the flow map's.

### Between-seed spread is the binding constraint

`orthrus` A@4 is 2.595 on one seed and 3.140 on the other, with an unchanged
objective. That half-token gap is larger than every contrast in the two
sections above except the self-correction one. Two seeds are enough to show
this; they are not enough to resolve effects of 0.1–0.4 tokens, and no number
of prompts fixes that — the variance is in training, not in decoding.

### What the audit changed

An adversarial review of the design — run blind, before it saw any number —
predicted most of this correctly and named the defects that produced the rest.
Four were real and all four inflated the earlier conclusions:

| defect | effect |
|---|---|
| SE over 120 cells when the masked configurations have 24 independent units | three of four "established" findings |
| position weights on the flow configurations only | the claimed parity was partly a weight contrast |
| `fixed_prior` not covering the restart draw | configurations unpaired on second-pass noise at `n>1` |
| cosine rebuilt on resume, LR back to ~79% of peak | the 4000-step point was systematically handicapped |

Two more were bugs in the self-correction term itself: the masked branch's
second forward was bidirectional, so the row predicting position `j` read the
committed draft token straight out of slot `j` — a copy shortcut on exactly the
full-weight positions — and training committed `K/2` positions where decoding
commits `round(K(k+1)/n)`.

A configuration trained with the *old* weight profile (`flow + self-corr, old
weighting`) is kept as a control on the fix itself. It is worse, but only
slightly and only on one seed: the copy-shortcut argument was right in sign and
small in size.

### Known limits### Known limits (what the next stage must fix)

- **One training seed per configuration.** The intervals cover decode noise, not
  training-run variance. Between-snapshot spread on an unchanged objective
  reaches 0.5 tokens — larger than every contrast in finding 3. Multiple seeds
  are what would move findings 2-3 from "trend plus t" to settled.
- **One eval dataset.** math500 only. gsm8k / humaneval / mbpp configs exist.
- **One scale.** 135M, `anchors_per_sequence=1`, `K=8`.
- `val_decode_prompts=0` in these runs, so no checkpoint was selected on
  `val/tpf`. Turn it back on for anything longer.

### Where things are written

| what | where | by |
|---|---|---|
| per-step scalars (`train/loss`, `loss/verify_kl`, `loss/selfcorrect_kl`, `loss/ec`, `loss/td`, `loss/accepted`) | `<output_dir>/lightning_logs/version_*/events.out.tfevents.*` | Lightning TensorBoard logger |
| validation decode (`val/tpf`, `val/acceptance_pos_*`) | same event file | `_maybe_decode_val` |
| checkpoints (FP32 DF head only; frozen backbone restored from HF) | `<output_dir>/last.ckpt`, `snap-*.ckpt` | `on_save_checkpoint` |
| one JSON row per eval run | `results/eval.jsonl` | `src/eval.py` |
| per-prompt eval detail | `results/eval-prompts.jsonl` | `src/eval.py` |
| bench cells (`A`, `tpf`, raw per-cell lists) | `results/{final_ab,longer,six_k,fill}.json` | the measurement driver |
| figures | `results/figures/*.png` | the figure script |

### Reproducing

Every configuration is `src/train.py` with a different set of loss weights; nothing else
differs. Common to all: `model=smollm2_135m train.block_size=8
train.anchors_per_sequence=1 data.batch_size=2 data.max_length=256
data.shuffle_buffer=64 trainer.precision=32`.

```bash
# masked                 (one component)
train.variant=orthrus
# masked + self-corr
train.variant=orthrus train.selfcorrect_kl_weight=1.0
# flow                   ([1] only)
train.variant=flowdraft_block_wise train.prior_type=discunif \
  train.verify_kl_weight=1.0 train.endpoint_weight=0.0 train.lambda=0.0 \
  train.terminal_time_fraction=1.0 train.teacher_chain_tail_weight=0.3 \
  'train.position_weights=[2.32,1.75,1.41,0.87,0.39,0.19,0.07]'
# flow + self-corr       (add to the above)
train.selfcorrect_kl_weight=1.0 train.selfcorrect_rounds=2 train.selfcorrect_s_min=0.0
# flow + self-corr + CFM (instead of the zeros above)
train.endpoint_weight=0.5 train.lambda=0.25 train.terminal_time_fraction=0.25 \
  train.lambda_ramp_steps=500
```

Longer horizons continue from a checkpoint with
`resume_from_checkpoint=<path> trainer.max_steps=6000
train.checkpoint_every_n_steps=2000 "train.checkpoint_name='snap-{step:07d}'"`
(the quotes matter — Hydra's override grammar rejects a bare `{`).

Acceptance is then read with `src/eval.py` per configuration and schedule
(`decode.jumps=1`, `'decode.jumps=[[0,1],[0.5,1]]'`, `decode.fixed_prior=true`),
one JSON row per run into `results/eval.jsonl`.

Two things worth carrying into any harness built on this: a run counts as
trained only when `global_step > 0` — Lightning writes `last.ckpt` on teardown
after an exception too, so the file's existence proves nothing — and readiness
is best polled from that checkpoint rather than from process exit, because
training finishes and the process then hangs indefinitely in pyarrow's
thread-pool destructor.

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

The ECLD target is stop-gradiented. Verifier alignment lives in `train.verify_kl_weight` (the pair the decode loop executes) and `train.selfcorrect_kl_weight` (the same alignment along the drafter's own jump schedule).

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
- **Objective** (`FlowDraft.compute_loss`): `loss = verify_kl_weight·verify_KL + selfcorrect_kl_weight·selfcorrect_KL + endpoint_weight·endpoint + λ·(4·EC + 2·TD)`
  - **verify KL** — block-wise inference alignment: `KL(sg(p_AR) ‖ π_{0,1}(x_0))` on the exact one-jump query used by decoding.
  - **endpoint** — `CE(x1, π_{t,t}(x_t))`: the paper's categorical VFM diagonal objective. The paper-faithful evaluation point is `train.anchor_point=trajectory`; `landing` is retained as an experimental option.
  - **AR KL** — optional `KL(sg(p_AR) ‖ π_{t,t})`, separately weighted and off by default because it is not part of the CFM objective.
  - **EC** — eq. (18) of *Categorical Flow Maps*: `CE(sg(π_{t,t}(X_{s,t}(x_s))), π_{s,t}(x_s))` — jumps learn from the diagonal at their own landing point; truth flows `x1 → π_{t,t} → π_{s,t}`.
  - **TD** — eq. (16): temporal drift `‖∂_t π_{s,t}‖²`.
  - Time pairs `(s, t)` per sample (`train.time_sampling`): `paper` (default: t~U, s~U[0,t]) | `triangle` | `sequential`.
- **Training geometries** (`train.variant`): `flowdraft` noises the full sequence; `flowdraft_block_wise` trains FlowDraft in the exact inference geometry; and `orthrus` uses Orthrus' single-step, dual-pass block-causal masked-diffusion geometry with no time conditioning. Both blockwise implementations can flatten several isolated width-K blocks into one drafter pass via `anchors_per_sequence`, sharing one full AR teacher/cache pass.
- **Decoding** (`FlowDraft.generate`): a width-K block contains one clean pending anchor plus K-1 fresh drafts produced in 1–few jumps, then ONE AR forward verifies the block. The previous cycle's correction/bonus token is never committed by its own pass: it rides as the clean in-block anchor and the next verify forward commits its K/V while scoring the drafts — **cycle cost = `jumps + 1` forwards** (TPF parity with the Orthrus convention). `temperature=0`: greedy verification, output **bit-identical** to `ar_generate`. `temperature>0` with Gumbel-coupled sampling (default): position-keyed Gumbel noise turns sampling into a deterministic argmax — the output is **bit-identical** to sampled `ar_generate` with the same seed. Uncoupled (`coupled=false`): Leviathan speculative sampling, lossless **in distribution**.

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
    │   └── orthrus.py             # Orthrus masked drafter, block-causal
    ├── preprocessor/df_processor.py   # tokenization + one-hot simplex endpoints
    ├── data/dataloaders.py        # streaming Dataset / collate / DataLoader;
    │                              #   EpochShuffled: repetitions in a new order (epochs)
    ├── configs/                   # hydra configs
    │   ├── train.yaml             # training entrypoint config
    │   ├── eval.yaml              # evaluation entrypoint config
    │   ├── model/                 # qwen3_1.7b (default) | qwen2_0.5b | smollm2_135m
    │   ├── data/                  # nemotron (training) | math500 (eval, unseen in training)
    │   └── experiment/            # one preset per task stage + additions:
    │                              #   orthrus | flowdraft_packed_blockwise
    │                                  ├── train.py                   # training entrypoint
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
                                                                   #   flowdraft_packed_blockwise
./hf-auth.sh uv run python src/train.py train.variant=flowdraft_block_wise   # ADDITION: inference geometry
```

Variants: `flowdraft` is the full-sequence flow-map objective; `orthrus` is the
paper-style block-causal Orthrus recipe (frozen AR cache plus independently
anchored masked blocks); and `flowdraft_block_wise` trains the flow-map
drafter in that inference geometry.
Knobs live in `configs/train.yaml`: `lambda`/`endpoint_weight`/`selfcorrect_kl_weight`/`lambda_ramp_steps`
(VFM/ECLD balance + staging), `anchor_point`, `time_sampling`,
`block_size`/`min_prefix`, `val_decode_prompts` (val-time decode -> `val/tpf`
curves + checkpoint monitor), `early_stop_patience`, optimizer, Lightning
`trainer.*`. Checkpoints store the FP32 DF head + its Adam moments; the frozen
backbone is never written. FP32 masters prevent late cosine-schedule updates
from disappearing through BF16 parameter rounding.

Checkpointing has three independent outputs:

- `<train.checkpoint_name>.ckpt` is an unconditional recovery snapshot every
  `checkpoint_every_n_steps` optimizer steps. These snapshots are all retained.
- `best-tpf-*.ckpt` contains the best validation states selected by fresh
  `val/tpf`; `checkpoint_save_top_k` controls how many are retained.
- `last.ckpt` is explicitly written at normal training completion (and
  best-effort on an exception), so it contains the terminal step even when that
  step is not a periodic checkpoint boundary.

An uncatchable process kill or a full filesystem cannot produce a final file.
The Russian guide is maintained separately in `README.ru.md`.

**Resume after interruption.** Use the newest periodic checkpoint after a hard
interruption, or `last.ckpt` after a normal/handled termination. Resume the full
Lightning state—DF weights, AdamW, cosine schedule, global step, and
callbacks—with:

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
| `train.variant` | `flowdraft` | which drafter to train: `flowdraft` (full-sequence CFM) \| `train.block_size` | 64 | total block width K: one clean anchor + K-1 drafted positions |
| `train.anchors_per_sequence` | 1 | number of isolated anchor+K blocks trained per packed sequence; packed block-wise FlowDraft defaults to 4 |
| `train.min_prefix` | 1 | shortest clean prefix before the training block |
| `train.respect_document_boundaries` | true | full-sequence FlowDraft isolates DF attention/losses by document; block-wise variants prevent drafted windows from crossing document boundaries |
| `train.lr` / `weight_decay` / `betas` | 1e-4 / 0.01 / [0.9, 0.95] | AdamW over the DF head only; `lr` is the PEAK of the schedule |
| `train.lr_schedule` | `cosine` | `cosine` (linear warmup → cosine decay to 0; needs a finite `trainer.max_steps` or `limit_train_batches`+`max_epochs`) \| `constant` |
| `train.warmup_ratio` | 0.05 | cosine only: fraction of total steps spent warming up |
| `train.time_sampling` | `paper` | how (s, t) pairs are drawn: `paper` \| `triangle` \| `sequential` |
| `train.lambda` | 1.0 | weight of the consistency part (4·EC + 2·TD) |
| `train.endpoint_weight` | 1.0 | weight of categorical VFM endpoint CE; 0 = endpoint-off ablation |
| `train.selfcorrect_kl_weight` | 0.0 | the multi-step term: the drafter's own jump schedule, each pass supervised by the frozen AR sweep over the pass before it |
| `train.verify_kl_weight` | 0.0 | direct block-wise `KL(p_AR ‖ π_{0,1})` on the exact one-jump inference pair |
| `train.lambda_ramp_steps` | 0 | staged distillation: lambda 0 → `lambda` over N steps; 0 = static |
| `train.anchor_point` | `trajectory` | where the anchor evaluates the diagonal: `trajectory` = π_{t,t}(x_t) \| `landing` = π_{t,t}(X_{s,t}(x_s)) |
| `train.checkpoint_name` | `flowdraft-{step:07d}` | checkpoint filename pattern — set your own per experiment (quote on CLI: `'train.checkpoint_name="my-run-{step:07d}"'`) |
| `train.checkpoint_every_n_steps` | 1000 | unconditional recovery snapshot interval in optimizer steps; all periodic snapshots are retained |
| `train.checkpoint_save_top_k` | 2 | how many best validation-metric checkpoints to retain |
| `train.best_checkpoint_name` | `best-tpf-{step:07d}` | filename pattern for metric-selected checkpoints |
| `train.final_checkpoint_name` | `last.ckpt` | terminal checkpoint, written independently of the periodic interval |
| `train.val_decode_prompts` / `val_decode_max_new` | 2 / 32 | run the real decode loop on N val prompts each validation → `val/tpf`, legacy prompt-mean `val/acceptance_decode`, pooled `val/decode/acceptance_pos_*`, and `val/decode/accepted_cycle_*`; 0 = off |
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
| `per_prompt_file` | `results/eval-prompts.jsonl` | prompt-level metrics and first-divergence diagnostics |
| `run_id` / `experiment_id` / `split_label` | null | optional result attribution; `experiment_id` can be shared across training seeds |
| `lossless_policy` | `assert` | canonical eager runs assert; separate SDPA throughput audits use `diagnose` |
| `data.truncation` | false | evaluate the complete rendered dataset sample; dataset `max_length` limits remain active during training |
| `decode.block_size` / `decode.jumps` | 8 / 1 | inference total width K (one anchor + 7 drafts at K=8) and refinement passes — knobs of EVERY variant |
| `decode.max_new_tokens` | 64 | tokens generated per prompt |
| `decode.n_prompts` | 64 | prompts taken from the dataset (100–200 for a paper table) |
| `decode.prompt_offset` | 0 | skip N usable prompts for reproducible disjoint development/test slices |
| `decode.prompt_len` | null | null = the full rendered prompt; int N = first N tokens only |
| `decode.temperature` / `top_k` / `top_p` | 0 / null / null | 0 = greedy; >0 = sampling |
| `decode.coupled` | true | T>0: Gumbel-coupled sampling — bit-exact vs AR |
| `decode.equiv_samples` | 0 | uncoupled only: N draws for the TV law-equivalence test; 0 = off |

**`model/*` — backbone** (`qwen3_1.7b` default; `qwen2_0.5b` is the brief's scale, `smollm2_135m` the bench):
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
| `flowdraft_packed_blockwise` | boundary-aware packed-2048 inference-geometry FlowDraft | `checkpoints/flowdraft-packed-blockwise/` |

Your own experiment (e.g. the `anchor_point` study) — override name and dir
so it gets its own shelf too:

```bash
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_packed_blockwise \
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

Dataset prompts (complete rendered samples by default; `decode.prompt_len=N`
for explicit N-token prefixes) are decoded twice — flow-draft vs plain AR —
and compared. Dataset `max_length` limits apply to training, not standalone
evaluation. Canonical evaluation uses eager attention and asserts greedy
losslessness **bitwise**, not by assumption. SDPA throughput audits should use
`lossless_policy=diagnose` and separate result files.

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
`eager` (canonical evaluation default) | `sdpa` (fused throughput audit) |
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
