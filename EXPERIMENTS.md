# Multi-step drafting: what was tested and what came out

Ten training runs at one seed, plus three of them replicated at two more seeds.
Measured on 460 tasks from six benchmarks — 56,000 per-prompt observations at
seed 42 across three horizons and four decode schedules, and 144 further
measurements across three seeds at the common 20k horizon.

---

## 1. Setup

A frozen autoregressive model decodes one token per forward pass. A lightweight
**drafter** proposes a block of `K−1 = 31` tokens at once; the frozen model
checks all of them in a single pass and accepts the longest prefix on which the
drafter's choice matches its own. The emitted text is **bit-identical** to plain
sequential decoding — verification guarantees it.

Two quantities, and they must not be conflated:

- **Accepted tokens** `A` — how many of the 31 proposals survive verification.
  This is drafting quality.
- **Tokens per forward** `TPF = (A + 1) / (n + 1)` where `n` is the number of
  drafter passes in a cycle. This is speed. Plain decoding gives exactly 1.

The drafter may take several **refinement passes** per cycle: propose, freeze the
positions it is most confident about, rewrite the rest, repeat. Each pass costs
one forward, so rising quality fights rising cost.

### Why acceptance is a conjunction

Position `j` is accepted only if every position before it was also accepted:
the verifier conditions on the tokens it has already committed. Hence

```
E[A] = Σ_{j=1..K−1} Π_{i≤j} a_i
```

where `a_i` is the per-position agreement probability. The derivative

```
∂E[A]/∂a_j = S_{j−1} · (1 + R_j),   S_j = Π_{i≤j} a_i,   R_j = a_{j+1}(1 + R_{j+1}),  R_{K−1} = 0
```

says what an improvement at position `j` is worth: the probability of ever
reaching it. At `a ≈ 0.8` the first position is worth 6.2 and the thirty-first
0.0015 — a factor of four thousand. This derivative is used to weight the loss
per position; it is verified against central differences to `1.4·10⁻⁷`.

### Two drafter parameterisations

- **Masking** — unpredicted positions carry a trainable placeholder vector.
  This is the Orthrus construction.
- **Continuous state** — a position carries a point on the vocabulary simplex,
  `x_s = (1−s)·x₀ + s·x₁`, interpolating a prior draw `x₀` and the answer `x₁`.
  The network parameterises a **flow map** `π_{s,t}(x_s)`: the distribution the
  state should reach by time `t`. The transport is
  `X_{s,t}(x) = x + γ·(π − x)` with `γ = (t−s)/(1−s)`.
  An intermediate position then carries a **graded** notion of "how settled this
  position is", which a placeholder cannot express.

---

## 2. The objective

### Continuous state

```
L = w_verify · KL( sg p_AR( · | ctx ) ‖ π_{0,1}(x₀) ) · u_j
  + w_self   · (1/r) Σ_{k=1..r} l( p_AR( · | ctx, argmax q_{k−1} ) , π_{s_k,1}(x_k) ) · v_j · u_j
  + w_end    · CE( x₁ , π_{t,t}(x_t) )
  + λ        · ( 4·EC + 2·TD )
```

with

```
q₀   = π_{0,1}(x₀)                        the jump the decode loop actually executes
x_k  = (1−s_k)·x₀′ + s_k·q_{k−1}          restart carrying the previous pass's draft
s_k  stratified in (s_min, 1), detached between passes
u_j  = ∂E[A]/∂a_j  ×  chain-validity gate
v_j  = 1 at the verifier's break position, `tail` elsewhere
EC   = CE( sg π_{t,t}(X_{s,t}(x_s)) , π_{s,t}(x_s) )
TD   = ‖∂_t π_{s,t}‖² · γ²
```

**First term — verifier alignment.** Match the frozen model's distribution on the
jump that ends every decode cycle. Its target does not depend on the drafter's
state, so this term alone cannot teach the drafter to *read* its input.

**Second term — multi-step training.** This is the addition under test. The
drafter proposes; one frozen forward over that proposal yields both a target
`p_AR(·|ctx, draft)` and the exact greedy verdict; the next pass is trained on
the state the decode loop actually visits. States are detached between passes:
the term asks for per-jump stationarity, not for a differentiable composition.

**Third and fourth — trajectory structure.** Endpoint likelihood on the diagonal
plus two consistency terms that make the family a trajectory rather than a set
of unrelated maps.

### Masking

```
L = KL( sg p_AR( · | ctx ) ‖ π(mask) ) · u_j
  + w_self · (1/r) Σ_{k=1..r} l( p_AR( · | ctx, d_{k−1} ) , π(state_k) ) · v_j · u_j
```

`state_k` freezes `keep_k = round((K−1)·(k+1)/(r+1))` positions at their chosen
tokens and re-masks the rest. The set is monotone and the frozen tokens never
change — the same sequence the decoder walks at `r+1` passes. At `K = 32, r = 2`
this is `[10, 21]` on both sides, verified by instrumenting both paths.

**Asymmetry worth naming.** In the continuous branch the restart times `s_k` are
*distributed*; in the masked branch the commit schedule is *fixed*. That is not
an oversight: a continuous state admits a distribution over entry points, a
placeholder does not.

### What the theory predicts about speed

If the map realised the Jacobi sweep `T(ctx,d)_j = argmax p_AR(·|ctx, d_{<j})`
exactly, then `n` chained passes would fix positions `1..n`, giving
`A_n ≥ min(n, K−1)`. Substituting into `TPF = (A+1)/(n+1)` gives **exactly 1**.
The lemma is therefore a **floor**, not a speedup mechanism: acceleration needs
`A_n > n` strictly, and the bound only gives `A_n ≥ n`. The measurements below
confirm this — no multi-step variant exceeds 1.

---

## 3. Comparison with the paper

Orthrus (arXiv 2605.12825) trains three projection matrices of the diffusion
attention — queries, keys and values — at block size 32, two epochs over 600k
examples, forward KL as the objective. Table 4 hyperparameters are reproduced
here exactly: sequence length 2048, 256 anchor blocks per sequence, block size
32, two epochs, peak LR 2·10⁻⁴ with cosine schedule and 5% warmup, gradient
clipping 1.0, global batch 128. The backbone differs — SmolLM2-135M instead of
Qwen3-8B — so absolute numbers are not comparable to theirs; contrasts within
this set are, which is why experiment 1 exists.

Their Table 3 reports that multi-step refinement **drops** throughput from 6.35
to 3.53 tokens per forward, concluding that single-step projection is optimal.

**The detail that matters:** their multi-step variant is trained by *randomly
masking 50% of block positions*, adapted from Fast-dLLM-v2 — not on the state
sequence its own decoding visits. The conclusion is drawn from a model that was
never taught multi-step refinement.

---

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
| multi-step training, masking | +0.223 | ± 0.021 | +46.4 | 0.0005 | 0.230 / 0.225 / 0.214 |

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

**Multi-step buys quality, not speed.** Only single-pass decoding clears 1.0.
The continuous state loses the least as passes grow (1.257 → 0.675 against
1.219 → 0.507), but it does not beat plain decoding either — exactly as the
prefix-fixing lemma predicts.

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

## 6. What remains open

- **Three seeds give `df = 2`.** All headline contrasts clear the 4.30 threshold
  comfortably, but the design is small; a fourth seed would halve the critical
  value.
- **Eight of ten runs were still improving at the step budget.** The horizon,
  not convergence, sets these numbers.
- **The self-correction term has no potential function.** `argmax` kills the
  target gradient and the detach kills the input gradient, so the sum is not the
  gradient of any scalar in `θ`. Convergence was observed on every curve; it is
  not guaranteed by anything.
- **Assumptions the code cannot remove** are listed in
  [ASSUMPTIONS.md](ASSUMPTIONS.md), including one that was measured and found
  **unsatisfied**: the additive time conditioning cannot gate the input Jacobian
  at any realistic amplitude, so verifier alignment and multi-step training do
  share capacity. The multiplicative fix built to relieve it showed no effect.
