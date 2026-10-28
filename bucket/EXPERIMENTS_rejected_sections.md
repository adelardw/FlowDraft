# Разделы, снятые из EXPERIMENTS.md 4 октября 2026

Сняты по указанию: в EXPERIMENTS.md остаются Orthrus, лучший эксперимент
(непрерывное состояние + многошаговое обучение) и предстоящие эксперименты.
Ниже — дословно всё, что ушло: отвергнутые гипотезы, абляции, их результаты,
прежние выводы и временный раздел статуса.

## Status, October 2026: what did not work, what replaced it, what runs next

This file was written at the SmolLM2-135M stage. The first table collects every
idea that did not pay off — on any backbone — next to what is used instead. The
second lists what runs next. Protocols, intervals and the command that
recomputes each number are in [docs/](docs/README.md): comparisons C1–C12,
findings E1–E15. The plan and the paper map are in [TODO.md](TODO.md).

### Did not work — and what is better

| Idea | What was measured | Better instead | Where |
|---|---|---|---|
| Flow-consistency terms (endpoint, EC, TD) | −0.114 accepted tokens [−0.150, −0.078], SmolLM2-135M, three seeds | the two-term objective `verify_kl + selfcorrect_kl`; the terms are in `bucket/` | §3.5 |
| Multiplicative time conditioning (input gate) | +0.027, p = 0.23 — no effect | additive time conditioning, the default | §6 |
| Freezing the value projection | −0.244 [−0.281, −0.207] | train Q, K and V | §4 |
| Adapting the output projection O on the continuous drafter | −0.140 ± 0.027 at one pass, Qwen3-0.6B, 10k | Q, K, V only. Re-tested at 100k, because the released Orthrus weights do train O | C2, E1, C10 |
| Four-pass schedule entering at `s = 0.34`, outside the trained range `[0.5, 1)` | −0.733 ± 0.055 against three passes, Qwen3-0.6B, 80k | n4v, entries `0 · 0.5 · 0.7 · 0.85`: 4.189, the best result of the campaign | C5, E9 |
| Idempotence term (`flowdraft_idem`) | inside the trained range +0.007 ± 0.025 (n1) and +0.018 ± 0.046 (n3) at 10k; at 50k lower internal KLs but the same decode acceptance. Stopped at step 50,651 | keep decode entries inside the trained range (n4v) — worth more than the term | C7, C8 |
| Extra decode passes for speed | the break-even `A(n+1) − A(n) > TPF(n)` fails at every transition, on every backbone | one pass for speed. Multi-step *training* is what raises the single pass: +0.360 ± 0.035 at 80k | E10, C3 |
| Taylor expansion of drafter attention, first and second order | 83–97% of the attention mass sits on keys with `|δ| > 1.5`; all 28 layers replaced gives acceptance 0.08 and 0.00 | a linear map fitted in closed form to the attention output: 8 layers for −6%, no training | E12, E13 |
| Choosing layers by how well their attention is approximated | Spearman with the cost of replacing the layer down to −0.70 — it picks the wrong layers | depth, and how far a layer's twin moved from AR, as priors; selection by function | E14 |
| Training-free linearization of the drafter: attention, MLP, whole layers, mixed | never faster than Orthrus; the best is parity | train the hybrid with distillation — the 100k runs below | E15, C11, C12 |
| Greedy layer selection on 40 states | 1.15× Orthrus on the selection states, 0.86× held out | rank components by single-replacement cost averaged over both drafters | C12 |

### Upcoming

| Experiment | What it decides | Status |
|---|---|---|
| 2×2 on Qwen3-0.6B at 100k: {Orthrus, best multi-step} × {full, hybrid diffusion view}, Q, K, V, O throughout. The hybrid view replaces 18 components with distilled linear maps (`qwen06_linear_base`) | whether multi-step training keeps its margin with O at a larger budget; whether a trained hybrid view is faster than the full one | full pair since 3 Oct; hybrid multi-step since 4 Oct; hybrid Orthrus starts when full Orthrus finishes |
| Two more seeds for the 0.6B Q, K, V pair, 10k each | between-seed intervals at 0.6B | queued |
| Released Orthrus 1.7B / 4B / 8B with their multi-pass remasking, n1–n3 | the break-even law at the paper's own scale | queued, inference only |
| AIME at a 2048-token budget for the released 1.7B | closes the Table 1 comparison on all six sets | queued |
| Anatomy probes on 4B and 8B | whether "depth predicts importance" holds with size | queued |
| Reduced-vocabulary drafting head | the output layer is 26% of a 0.6B drafting pass; a frequency-ranked head could cut about 20% of it | idea |
| TV distance between draft and target on finished checkpoints | how much sampled decoding loses to near-one-hot drafts | idea, no quota needed |
| On-policy survival weights instead of the static acceptance profile | single-pass acceptance | idea |

---

### Two drafter parameterisations

- **Masking** — unpredicted positions carry a trainable placeholder vector.
  This is the Orthrus construction.
- **Continuous state** — a position carries a point on the vocabulary simplex,
  interpolating a prior draw and the answer, so an unfinished position expresses
  *how settled it is* rather than merely "unknown". Section 2 makes this precise.

Every experiment is named `<backbone>_<method>` with `_multistep` added when
the drafter is trained on its own refinement procedure, so
`qwen_flowdraft_multistep` is the continuous state with multi-step training on
Qwen3-1.7B. The same experiment carries the same name on both backbones:

| experiment | SmolLM2-135M config | Qwen3-1.7B config |
|---|---|---|
| Orthrus, reproduced | `smollm_orthrus` | `qwen_orthrus` |
| masked + multi-step | `smollm_orthrus_multistep` | `qwen_orthrus_multistep` |
| continuous, ablation | `smollm_flowdraft` | `qwen_flowdraft` |
| continuous + multi-step | `smollm_flowdraft_multistep` | `qwen_flowdraft_multistep` |

No name says "baseline": there is exactly one baseline here and it is Orthrus.
`*_flowdraft` is an **ablation** — FlowDraft with the multi-step term switched
off — and calling it a baseline would have asserted something untrue.

---

### 3.2 Masked plus multi-step training — `*_orthrus_multistep`

Still $d_\theta$, still no $(s,t)$. What changes is the **state argument**: the
second term feeds block states the decoder actually visits.

```math
\mathcal{L} \;=\;
\underbrace{\mathrm{KL}\Big(\mathrm{sg}\,p_{\mathrm{AR}}(\cdot\mid\mathrm{ctx}) \,\Big\|\, d_\theta(\cdot\mid\mathrm{ctx}, M_0)\Big)\cdot u_j}_{\text{verifier alignment}}
\;+\;
w_{\text{self}}\cdot
\underbrace{\frac{1}{r}\sum_{k=1}^{r}
\ell\Big(p_{\mathrm{AR}}\big(\cdot\mid\mathrm{ctx},\,\hat d_{k-1}\big),\;\; d_\theta(\cdot\mid\mathrm{ctx}, M_k)\Big)\cdot v_j\, u_j}_{\text{multi-step}}
```

where $\hat d_{k-1}$ is the token sequence proposed at pass $k-1$, and $M_k$
freezes its most confident slots:

```math
\mathrm{keep}_k \;=\; \mathrm{round}\!\Big((K-1)\cdot\frac{k+1}{r+1}\Big),
\qquad
M_k = \big\{\text{slots in } \mathrm{keep}_k \text{ hold } \hat d_{k-1}\text{'s tokens; the rest hold } \texttt{[MASK]}\big\}
```

```
  pass 0   │ anchor │ MASK │ MASK │ MASK │ ... │ MASK │   M₀, propose all 31
  pass 1   │ anchor │  t₄  │ MASK │  t₉  │ ... │ MASK │   M₁, 10 slots frozen
  pass 2   │ anchor │  t₄  │  t₇  │  t₉  │ ... │ MASK │   M₂, 21 slots frozen
```

The committed set is monotone and frozen tokens never change — the same
sequence the decoder walks at $r+1$ passes. At $K = 32$, $r = 2$ this is
$[10, 21]$ on both sides, verified by instrumenting both paths.

**A slot here is binary.** It is either `[MASK]` or a hard token. There is no
way to say "this slot is 70% settled", which is exactly what a time index would
express.

---

### 3.3 Continuous state, ablation — `*_flowdraft`

Now the indices appear. Only one pair is ever used: $(s,t) = (0,1)$ — from a
pure prior draw to the answer.

```math
\mathcal{L} \;=\; w_{\text{verify}}\cdot
\mathrm{KL}\Big(\mathrm{sg}\,p_{\mathrm{AR}}(\cdot\mid\mathrm{ctx})\;\Big\|\;\pi^\theta_{\,0,\,1}(x_0)\Big)\cdot u_j
```

Read the subscripts: input at time $s = 0$ (pure noise, no information about the
answer), output aimed at time $t = 1$ (the answer itself). This is the first refinement pass
of every decode cycle. The rest of the family — every $\pi^\theta_{s,t}$ with $s > 0$ —
receives no gradient from this term at all.

---

### 3.5 Trajectory-structure terms — measured, rejected

```math
\mathcal{L}_{\mathrm{CFM}} \;=\; w_{\text{end}}\cdot \mathrm{CE}\big(x_1,\,\pi^\theta_{t,t}(x_t)\big)
\;+\; \lambda\Big(4\,\mathrm{EC} + 2\,\mathrm{TD}\Big),
```
```math
\mathrm{EC} = \mathrm{CE}\Big(\mathrm{sg}\,\pi^\theta_{t,t}\big(X_{s,t}(x_s)\big),\;\pi^\theta_{s,t}(x_s)\Big),
\qquad
\mathrm{TD} = \big\|\partial_t \pi^\theta_{s,t}\big\|^2\,\gamma^2
```

These make the family a trajectory rather than a set of unrelated maps. Measured
at `−0.114` accepted tokens `[−0.150, −0.078]` — significantly harmful — and
moved to `bucket/`. Note the `4:2` ratio is only dimensionless for a unit time
parameterisation: `EC` is in nats and invariant to reparameterising `t`, while
`TD` carries $1/\mathrm{time}^2$.

## 4. Results

All numbers at the common 20k horizon, three refinement passes unless stated.
Intervals in this section use the **training seed** as the unit of observation
(three seeds, `df = 2`, `t₀.₉₇₅ = 4.30`) — they describe the spread of the
*method*, not of one trained model.

### The two headline claims

| contrast | Δ accepted | 95% CI | t | p | per seed |
|---|---|---|---|---|---|
| **multi-step training, continuous state** | **+1.138** | ± 0.109 | +45.0 | 0.0005 | 1.175 / 1.151 / 1.090 |
| **best vs reproduced Orthrus** | **+0.835** | ± 0.085 | +42.3 | 0.0006 | 0.870 / 0.834 / 0.802 |
| continuous state vs masking, same objective | +0.612 | ± 0.065 | +40.6 | 0.0006 | 0.640 / 0.610 / 0.588 |
| masked + multi-step vs reproduced Orthrus | +0.223 | ± 0.021 | +46.4 | 0.0005 | 0.230 / 0.225 / 0.214 |

**The last row is not an isolated mechanism, and it used to be labelled as one.**
It read *multi-step training, masking*, which claims the multi-step term alone
accounts for it. It does not. `smollm_orthrus_multistep` differs from the
reproduced baseline in three ways at once: it adapts the output projection as
well as Q, K and V, it carries the measured acceptance profile, and its
chain-tail weight is 0.3 against the baseline's 1.0.

Against a control matched on all three — masked state, four projections, the
same weights, no multi-step term — the multi-step term is worth **+0.046**
(positive on all six benchmarks, +0.015 to +0.085) rather than +0.223. **Four
fifths of that row is the bundle, not the mechanism.** The control was trained
and measured at one seed only, so the correction states a size, not an interval;
the +0.223 above keeps its three-seed interval because the comparison it makes —
against the published baseline — is the one the seeds were run for.

The other three rows are matched. *Multi-step training, continuous state*
differs in `selfcorrect_kl_weight` and `selfcorrect_s_min` and in nothing else;
*continuous state vs masking* holds projections, profile and tail fixed and
varies the state; *best vs reproduced Orthrus* is a method against a published
baseline, which is what it says.

The second scale point makes the same point from the other side: on Qwen3-0.6B,
training the continuous multi-step run on Q, K, V alone — the baseline's own
projection set — makes it **better**, not worse. The output projection was
costing acceptance while making every comparison against the baseline
unmatched. See the Qwen3-0.6B section of [README.md](README.md).

### Growth from one pass to four

| experiment | Δ | 95% CI | t | p | per seed |
|---|---|---|---|---|---|
| **continuous + multi-step** | **+0.893** | ± 0.085 | +45.1 | 0.0005 | 0.925 / 0.897 / 0.857 |
| masked + multi-step | +0.070 | ± 0.014 | +21.6 | 0.0021 | 0.068 / 0.066 / 0.077 |
| Orthrus, reproduced | +0.071 | ± 0.019 | +15.8 | 0.0040 | 0.069 / 0.064 / 0.079 |
| **continuous, no multi-step** | **−0.563** | ± 0.526 | −4.60 | 0.044 | −0.337 / −0.595 / −0.757 |

The gain from refinement is **thirteen times larger** with a continuous state,
and without the training the same architecture *loses* acceptance on all three
seeds.

### Throughput, averaged over seeds

| experiment | 1 pass | 3 | 4 |
|---|---|---|---|
| masked + multi-step | **1.326** | 0.681 | 0.550 |
| continuous + multi-step | 1.257 | **0.824** | **0.675** |
| Orthrus, reproduced | 1.219 | 0.628 | 0.507 |
| continuous, no multi-step | 1.245 | 0.554 | 0.394 |

**Separate the training term from the decode schedule — they pull opposite
ways.** Multi-step *training* raises throughput; extra *decode* passes lower it.
Read down the first column, where every method is fastest: multi-step training
is worth **+0.107 tokens per forward over Orthrus (+8.8%)** on the masked
drafter and **+0.038 (+3.1%)** on the continuous one, both significant with the
seed as the unit (`t = 129.5` and `t = 6.9`, `n = 3`). The best throughput in
the study is masked + multi-step at a single pass, 1.327 against 1.219.

Read across a row instead and the schedule takes it back: a cycle of `n` passes
spends `n+1` forwards while acceptance grows slower than `n`, so nothing beyond
one pass clears 1.0. The continuous state loses the least (1.257 → 0.675 against
1.219 → 0.507), which is precisely why its acceptance advantage — the largest
effect in this study — does not become speed. This is what the prefix-fixing
lemma predicts: `TPF = 1` is a floor, not a mechanism.

**Wall-clock does not follow, and this bench cannot settle it.** At one pass the
same runs give 1.211× plain decoding for masked + multi-step against 1.209× for
Orthrus — the per-forward gain vanishes in seconds, because a 135M forward on
MPS is dominated by fixed overhead. Tokens per forward is the
hardware-independent quantity; converting it to wall-clock needs the Qwen3-1.7B
run, which has not been done.

### A reversal at one pass

At a single pass, **masking wins**: −0.148 ± 0.047 in favour of the placeholder
(`p = 0.0055`). The advantage of a continuous state appears only with refinement
and grows with it: **−0.148 → +0.612 → +0.675** at one, three and four passes.

### Between-seed stability

| experiment | σ at 1 pass | at 3 | at 4 |
|---|---|---|---|
| Orthrus, reproduced | 0.004 | 0.001 | 0.001 |
| masked + multi-step | 0.005 | 0.008 | 0.004 |
| continuous + multi-step | 0.007 | 0.032 | 0.043 |
| **continuous, no multi-step** | 0.005 | 0.013 | **0.228** |

Untrained multi-step refinement is not merely worse, it is **unpredictably**
worse: at four passes the three seeds give 1.227 / 0.933 / 0.778. Every other
run stays within 0.05.

### Single-seed ablations (prompt-level intervals, seed 42 only)

| ablation | Δ | 95% CI | verdict |
|---|---|---|---|
| multiplicative time conditioning | +0.027 | [−0.006, +0.060] | not significant (p = 0.23) |
| flow-consistency terms | −0.114 | [−0.150, −0.078] | significantly harmful |
| freezing the value projection | −0.244 | [−0.281, −0.207] | significantly harmful |
| equal position weights, continuous | −0.043 | [−0.074, −0.012] | marginal (p = 0.022) |
| equal position weights, masked | −0.005 | [−0.022, +0.012] | no effect (p = 0.57) |

---

## 5. Conclusions

**The paper's conclusion is correct for its own architecture and does not
generalise.** With masking, acceptance barely responds to refinement passes
(+0.070) whether or not the model was trained on them. With a continuous state
trained on its own procedure, it grows by +0.893.

**Multi-step refinement must be trained, or it hurts.** The identical
architecture without that term loses 0.563 tokens going from one pass to four,
and the loss varies wildly across seeds.

**The continuous state is load-bearing.** Holding the objective fixed and
swapping the placeholder for a simplex point is worth +0.612.

**No speedup.** Throughput falls monotonically; nothing multi-step beats plain
autoregressive decoding. What multi-step buys is draft quality at a fixed block
width.

**Three ideas did not survive:** multiplicative time conditioning, flow
consistency terms, and freezing the value projection.

---

### Open

- **Three seeds give `df = 2`.** Every headline contrast clears the 4.30
  threshold comfortably, but the design is small; a fourth seed would halve the
  critical value.
- **Eight of ten runs were still improving at the step budget.** The horizon,
  not convergence, sets these numbers. *Update, September 2026:* on Qwen3-0.6B
  the two arms that carry the comparison went to 80k steps, and the margin grew
  rather than shrank (n1 +0.243 → +0.360, n3 +1.356 → +1.423; README, 80k
  section). A 100k pair is running.
- **`verify_kl` was never ablated.** At one refinement pass it does all the work
  (the multi-step term adds +0.018); at three or four the roles invert. Whether
  the term is still needed once multi-step training covers the restarts has not
  been measured — only argued, and the first decode pass does start at `s = 0`
  where no draft exists yet.
- ~~**Nothing has run on CUDA or on Qwen.**~~ *Resolved:* Qwen3-0.6B trained on
  a T4 (CUDA) to 80k steps, and the released Orthrus-Qwen3-1.7B was measured in
  this harness on a T4, reproducing the paper's Table 1 on four of six sets
  (docs E11). Still single-device: collective operations and the rank seed
  offset are no-ops on one device. The sparse FlexAttention path has since run
  on CUDA in the 0.6B campaigns; before that it never executed — though its
  mask was checked against the dense one it
  replaces over 4,981 (query, key) pairs across six block geometries and both
  in-block causality settings, with zero disagreement, so the two differ in
  speed and not in what they compute.
