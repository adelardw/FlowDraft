# Разделы, снятые из README 10 августа 2026

Сняты не потому, что неверны, а потому что перекрыты более поздним и
более сильным замером. Числа ниже получены на 6000 шагах и двух сидах;
кампания августа мерила те же контрасты на 20000 шагах и трёх сидах, и
именно её числа стоят в EXPERIMENTS.md. Держать в README два слоя
результатов, которые расходятся, — значит заставлять читателя гадать,
какой из них текущий.

Пресеты, на которых построен раздел «Experiments (task stages)», лежат
здесь же: `orthrus.yaml`, `flowdraft_packed_blockwise.yaml`.

---

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
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_packed_blockwise
#   the PRESET is flowdraft_packed_blockwise; flowdraft_block_wise is the
#   train.variant it sets — passing the variant name to +experiment= fails
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


---

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


# Разделы, снятые из README 26 октября 2026

Исходный бриф SMILES 2026: драфтер на категориальных картах потока
(Categorical Flow Maps) с дистилляцией согласованности. Замеры его не
подтвердили: члены согласованности стоили −0.114 принятого токена и убраны
(`bucket/cfm_terms`); при одном проходе непрерывное состояние не лучше маски,
а выигрыш даёт член обучения на собственной цепочке уточнения. README теперь
описывает проект как исследование целевой функции драфтера внутри Orthrus;
ниже — разделы брифа как они стояли.

## Key idea: a Categorical Flow Map drafter

*The pitch of the original project brief. Measured since: the drafter is
mean-field, a continuous state alone does not beat masking at one pass, and
what pays is training it on its own refinement chain — see the results
sections.*

- **Categorical Flow Maps** [Roos et al., 2026] learn the *integrated, correlated* endpoint distribution on the simplex and generate in **one or few jumps**.
- Use it as the drafter: a **higher-fidelity joint proposal** over the block — at the **same pass count**.
- Verification is unchanged → output stays **strictly lossless**; the drafter only affects *speed*, never *quality*.
- **Novelty:** a flow-map drafter inside Orthrus, trained on its own refinement chain against the frozen verifier. The flow-map consistency losses it started from were measured, found harmful and dropped ([EXPERIMENTS.md](EXPERIMENTS.md) §8).

**Why it matters**

1. **Efficiency** — higher acceptance length = higher throughput, for free.
2. **Fidelity** — speedup with **zero** quality loss (verification guarantees it).
3. **Foundations** — connects flow-map distillation to fast, faithful LLM inference.

## CFM training, in brief

*Where the project started. The self-consistency part below was measured, cost
−0.114 accepted tokens, and was removed; the objective in use is under
[Method](#method).*

The drafter learns two complementary parts of a categorical flow map:

- **Endpoint inference — *what endpoint belongs to the trajectory*.** The diagonal predictor is trained by categorical VFM against the clean endpoint used to construct the interpolant.
- **Self-consistency — *how to jump*.** The reliable diagonal prediction at a transported state teaches the harder long-jump predictor through ECLD.

The ECLD target is stop-gradiented. Verifier alignment lives in `train.verify_kl_weight` (the pair the decode loop executes) and `train.selfcorrect_kl_weight` (the same alignment along the drafter's own jump schedule).

The AR model remains frozen throughout. In paper-faithful CFM training it supplies the cached prefix for the block-wise geometry and validation targets; at inference it verifies every proposal, which is what guarantees losslessness.

## Goals

*The original project brief. What was kept and what was measured and dropped is
in [EXPERIMENTS.md](EXPERIMENTS.md) §8.*

1. **Reproduce Orthrus** (frozen AR + masked-diffusion drafter, shared KV cache, lossless loop) at a tractable scale.
2. **Implement a flow-map drafter** (simplex endpoint head, 1–few jumps).
3. **Develop the dual distillation objective** (AR-teacher distribution + flow-map consistency).
4. **Evaluate & compare:** AR baseline vs. masked-diffusion Orthrus vs. flow-map drafter — on acceptance length, TPF, and throughput — all verified lossless.

## Expected deliverables

*From the same brief; the objective in use is under [Method](#method).*

1. Reproduction of the Orthrus lossless parallel decoder (masked-diffusion drafter).
2. Implementation of the **Categorical Flow Map drafter** + dual distillation training.
3. Evaluation: acceptance-length / TPF / throughput comparison, with verified losslessness and **block-size / jump-count ablations**.
