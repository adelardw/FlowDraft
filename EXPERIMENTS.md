# FlowDraft: the baseline, the method, and what runs next

Two drafters are compared throughout: **Orthrus**, the published baseline, and
**the method** — a continuous drafting state trained on its own refinement
procedure (`*_flowdraft_multistep`). Measured at two scale points: SmolLM2-135M
(20k steps, three seeds) and Qwen3-0.6B (10k and 80k steps, one seed), on 460
tasks from six benchmarks, every greedy decode checked bitwise against plain
autoregressive decoding.

§7 lists the runs in progress and queued, with their mathematics. Ideas that
were measured and dropped appear once, in §8; their full sections moved to
`bucket/`. Per-number protocols and the commands that recompute them are in
[docs/](docs/README.md).

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

### Two drafters

- **Masking** — unpredicted positions carry a trainable placeholder vector.
  This is Orthrus, the baseline.
- **Continuous state** — a position carries a point on the vocabulary simplex,
  interpolating a prior draw and the answer, so an unfinished position expresses
  *how settled it is* rather than merely "unknown". Section 2 makes this precise.
  The method trains it on its own refinement procedure.

Configurations, named as in the README (`_multistep` marks training on the
drafter's own refinement):

| | Orthrus, the baseline | The method |
|---|---|---|
| SmolLM2-135M — 20k steps, three seeds | `smollm_orthrus` | `smollm_flowdraft_multistep` |
| Qwen3-0.6B, Q, K, V — 10k and 80k steps | `qwen06_orthrus` | `qwen06_flowdraft_multistep_qkv` |
| Qwen3-0.6B, Q, K, V, O — 100k steps, running | `qwen06_orthrus_qkvo` | `qwen06_flowdraft_multistep` |

There is exactly one baseline here and it is Orthrus; no configuration name
says "baseline".

---

## 2. Notation: what K is, and where the simplex lives

**`K` is the block width in tokens**, and nothing else. One decoding cycle emits
a block of `K` positions: position 0 is a **clean anchor** — a real token the
verifier already committed — and positions `1 … K−1` are **drafted**. With
`K = 32` the drafter proposes 31 tokens per forward pass. `K` is a property of
the decoding geometry; it is not a parameter of the flow-matching formulation
and does not appear in any loss term.

The flow map itself is **per position**: each drafted slot carries its own point
on the vocabulary simplex and its own map. There is no "block-size" notion
inside categorical flow matching — the block is how many independent slots are
run in parallel through the same forward pass. `K` sets the acceptance ceiling
(you cannot accept more than `K−1` tokens per cycle) and the cost of a cycle,
not the shape of the objective.

### What the drafter is fed, position by position

Both parameterisations receive the same block layout. They differ only in what
occupies the drafted slots.

```
                 slot 0      slot 1      slot 2     ...    slot K-1
                ┌────────┬───────────┬───────────┬─────┬───────────┐
  masking       │ anchor │  [MASK]   │  [MASK]   │ ... │  [MASK]   │
                │ token  │  vector   │  vector   │     │  vector   │
                └────────┴───────────┴───────────┴─────┴───────────┘
                ┌────────┬───────────┬───────────┬─────┬───────────┐
  continuous    │ anchor │   x_s     │   x_s     │ ... │   x_s     │
  state         │ token  │  simplex  │  simplex  │     │  simplex  │
                └────────┴───────────┴───────────┴─────┴───────────┘
                    ↑          └──── these are the drafted positions ────┘
              real token,
              never masked,
              never gated
```

The anchor is the only clean token in the block: the KV cache is cropped before
the block, so that token exists **only** in the anchor row. Attention is
bidirectional inside the block and causal into the cached prefix.

The simplex point interpolates a prior draw and the answer,

```math
x_s \;=\; (1-s)\,x_0 \;+\; s\,x_1 ,\qquad x_0 \sim \text{prior},\quad x_1 = \text{one-hot answer}
```

and the network parameterises a **flow map** — the distribution the slot should
reach by time `t`:

```math
\pi^\theta_{s,t}(x_s) \;=\; \mathrm{softmax}\big(f_\theta(x_s,\,s,\,t)\big),
\qquad
X_{s,t}(x) \;=\; x + \gamma\,(\pi^\theta_{s,t}(x) - x),
\qquad
\gamma = \frac{t-s}{1-s}
```

At `(s,t) = (0,1)` we get `γ = 1` and the jump is `π` itself — that is the
single pair the decode loop executes. A masked slot has no `s`: it is either
"unknown" or "committed", with nothing in between. That difference is the whole
subject of this study.

### Why acceptance is a conjunction, and what the weights are

Position `j` is accepted only if every earlier position was accepted too:

```math
\mathbb{E}[A] \;=\; \sum_{j=1}^{K-1} \prod_{i \le j} a_i ,
\qquad
\frac{\partial \mathbb{E}[A]}{\partial a_j} \;=\; S_{j-1}\,(1 + R_j),
\quad S_j = \prod_{i\le j} a_i,
\quad R_j = a_{j+1}(1+R_{j+1}),\; R_{K-1}=0
```

This derivative is the per-position weight `u_j`. At `a ≈ 0.8` it runs from 6.2
at the first position to 0.0015 at the thirty-first — a factor of four thousand.
Verified against central differences to $1.4 \times 10^{-7}$.

---

## 3. The losses

### Notation: the two drafters are different objects

They are written with different letters throughout, because they are not the
same kind of function.

| | masked drafter | flow map |
|---|---|---|
| symbol | $d_\theta$ | $\pi^\theta_{s,t}$ |
| what it is | **one** map | a **two-parameter family** of maps |
| input | block state $M$ — every slot is either `[MASK]` or a committed token | simplex point $x_s$ — every slot is a distribution part-way between prior and answer |
| indices | **none** | $s$ = time of the *input* state, $t$ = time the *output* should reach |
| slot values | discrete: unknown / committed | continuous: $x_s = (1-s)x_0 + s\,x_1$ |
| what "progress" means | how many slots are committed | how far along $s$ every slot is, individually |

```math
d_\theta\big(\cdot \mid \mathrm{ctx}, M\big) \;:\; \text{block state} \longrightarrow \text{distribution over the vocabulary}
```

```math
\pi^\theta_{s,t}\big(x_s\big) \;:\; \text{simplex point at time } s \longrightarrow \text{distribution the slot should hold at time } t
```

The masked drafter has no $s$ because its input carries no notion of partial
progress, and no $t$ because its output is always "the answer" — there is no
intermediate target to aim at. **That absence is the subject of this study.**

Shared symbols:

| symbol | meaning |
|---|---|
| $p_{\mathrm{AR}}(\cdot\mid\mathrm{ctx})$ | frozen verifier's distribution, stop-gradient throughout |
| $u_j$ | $\partial\mathbb{E}[A]/\partial a_j$ times the chain-validity gate |
| $v_j$ | 1 at the verifier's break position, `tail` elsewhere |
| $r$ | refinement passes trained per step (2 here) |
| $M_k$ | masked block state after $k$ commits |
| $x_k$ | simplex block state at restart $k$ |

---

### 3.1 Orthrus, the baseline — `*_orthrus`

One term. Every drafted slot holds the mask vector, so the state is $M_0$ — all
slots unknown. **No indices anywhere**: a single map, a single target.

```math
\mathcal{L} \;=\; \mathrm{KL}\Big(\,\mathrm{sg}\;p_{\mathrm{AR}}(\cdot\mid \mathrm{ctx})\;\Big\|\;d_\theta(\cdot\mid\mathrm{ctx}, M_0)\Big)
```

No position weights, no chain gate, projections $W_Q, W_K, W_V$ only. Nothing
depends on the drafter's own output, so multi-step refinement is available at
decode time but is never trained.

**Fidelity to the paper.** Checked against the published training design line by
line. What matches: $B$ sampled anchor positions per batch with contiguous blocks
of width $K$; the corruption rule (anchor kept visible, remaining $K-1$ slots
replaced by `<mask>`); forward KL against the frozen AR head's full distribution;
gradients confined to the diffusion module; and the dual-pass block mask, whose
two clauses — causal AR context $\mathbf 1[k \lt L]\cdot\mathbf 1[k\le a_b-1]$ and
bidirectional-within-block $\mathbf 1[k\ge L]\cdot\mathbf 1[\lfloor q/K\rfloor=\lfloor (k-L)/K\rfloor]$ —
appear verbatim in both the sparse FlexAttention path and the dense fallback.

**Geometry.** Every trained run here, Orthrus included, uses the bench
geometry: block size $K = 32$, one anchor block per sequence, context 256,
effective batch 16. The paper's Table 4 ($L = 2048$, 256 anchor blocks, two
epochs over 600K examples, global batch 128) is reproduced in the Qwen3-1.7B
presets (`qwen_orthrus`), which were never trained.

**Which projections.** The paper's text names $W_Q, W_K, W_V$, and `*_orthrus`
follows the text. The released weights (`chiennv/Orthrus-Qwen3-1.7B`) train
more: the twin of the output projection moved from its AR initialisation by
0.34 in relative norm — as much as Q, K and V (0.32, 0.35, 0.31) — with
full-rank updates, while the per-head norms barely moved (0.001–0.002). The
paper's "approximately 16% of the total model" agrees with the release rather
than with the text once the total includes the diffusion module: for Qwen3-8B,
Q, K, V would be 10.0% and Q, K, V, O 15.6%. `qwen06_orthrus_qkvo` follows the
release (§7.1; docs/evidence.md, entry E1).

**The block yields $K$ tokens per cycle, of which the drafter supplies $K-1$.**
The paper builds its block "by taking the current anchor token $x_t$ and
concatenating it with $K-1$ `<mask>` tokens", and supervises the objective "over
all masked positions" — of which there are $K-1$, the anchor not being masked.
The anchor is itself a generated token, emitted by the AR head on the previous
cycle, so a cycle produces one AR token plus up to $K-1$ drafted ones and costs
"exactly two forward passes". Our implementation is the same object: `drafted =
block_size - 1`, the anchor is the pending AR token, and verification is the one
AR forward. The bound $k=1,\dots,K$ written on the sum in the objective reads as
$K$ terms, but the prose fixes the meaning twice, and the inference construction
agrees with the prose.

---

### 3.2 The method: a continuous state trained on its own refinement — `*_flowdraft_multistep`

The main result. The second term reaches **into the family**: it trains
$\pi^\theta_{\,s_k,\,1}$ at several interior $s_k$, which is precisely what the
masked parameterisation cannot express.

```math
\mathcal{L} \;=\; w_{\text{verify}}\cdot
\underbrace{\mathrm{KL}\Big(\mathrm{sg}\,p_{\mathrm{AR}}(\cdot\mid\mathrm{ctx}) \,\Big\|\, \pi^\theta_{\,0,\,1}(x_0)\Big)\cdot u_j}_{\text{trains one member: } s=0}
\;+\;
w_{\text{self}}\cdot
\underbrace{\frac{1}{r}\sum_{k=1}^{r}
\ell\Big(p_{\mathrm{AR}}\big(\cdot\mid\mathrm{ctx},\,\arg\max q_{k-1}\big),\;\; \pi^\theta_{\,s_k,\,1}(x_k)\Big)\cdot v_j\, u_j}_{\text{trains members at } s_1,\dots,s_r}
```

where the draft fed to the next round is defined by **one recursion**, run for
$k = 0, 1, \dots, r$:

```math
x_k \;=\; (1-s_k)\,x_0^{(k)} \;+\; s_k\,q_{k-1},
\qquad
q_k \;=\; \mathrm{sg}\;\pi^\theta_{\,s_k,\,1}(x_k),
\qquad
s_0 = 0,
\qquad
s_1 < \dots < s_r \ \text{stratified in } (s_{\min}, 1)
```

Reading it at $k=0$ gives $x_0 = x_0^{(0)}$ and $q_0 = \mathrm{sg}\,\pi^\theta_{\,0,\,1}(x_0)$ —
so the **first term of the loss and $q_0$ are the same forward pass**, scored
once with gradient and reused once detached. Each $x_0^{(k)}$ is an
*independent* prior draw, and the $\mathrm{sg}$ is why round $k+1$ receives the
previous draft as **data rather than as a gradient path**: without it the
$r$ rounds would collapse into one long chain through $\theta$.

**What $\arg\max$ does here.** $q_{k-1}$ holds one distribution over the
vocabulary per drafted position, and the $\arg\max$ is taken along the
vocabulary axis, *independently at each position* — turning the draft from
distributions into concrete tokens, exactly the block the drafter would propose
at decode. Conditioning on tokens rather than on $q_{k-1}$ itself is forced by
what acceptance means: position $j$ is accepted iff its token equals the
verifier's own $\arg\max$ given the tokens before it, so a target conditioned on
a soft mixture would be conditioned on something the verifier never sees. This
is also the only term whose target depends on the drafter's state at all —
every other teacher term aims at $p_{\mathrm{AR}}(\cdot\mid\mathrm{corpus})$,
which the drafter's output does not enter. The price is that $\arg\max$ has zero
derivative almost everywhere, so **no gradient reaches $\theta$ through the
target**; together with the $\mathrm{sg}$ on $q_{k-1}$ that makes this term a
DAgger step rather than the gradient of anything — see the assumptions table
in §9.

Note that $t = 1$ in **every** term: the drafter is always asked for the answer,
never for an intermediate distribution. What varies is $s$ — how far along the
input already is. Since $x_k$ mixes a fresh prior draw with $q_{k-1}$, the value
of $s_k$ literally sets how much of the previous draft survives into the next
input.

```
  pass 0   │ anchor │  x₀  │  x₀  │  x₀  │ ... │  x₀  │   s = 0,   pure prior
  pass 1   │ anchor │  x₁  │  x₁  │  x₁  │ ... │  x₁  │   s = s₁,  x₁ = (1-s₁)x₀' + s₁q₀
  pass 2   │ anchor │  x₂  │  x₂  │  x₂  │ ... │  x₂  │   s = s₂,  x₂ = (1-s₂)x₀' + s₂q₁
```

**Every slot moves at every pass** — nothing is frozen, because a simplex point
expresses "mostly settled" without committing. That is the structural difference
from a masked drafter, where a slot is either masked or fixed and the only thing that can
change between passes is *which* slots are fixed.

States are detached between passes: the term asks for per-jump stationarity,
not for a differentiable composition through $r$ passes.

**$s_{\min} = 0.5$, not 0.** The draft is recoverable from $x_k$ only when its
component dominates the prior's:

```math
s \;>\; \frac{1}{1 + q_{\max} - q_{r}} \;\approx\; \frac{1}{1+q_{\max}}
```

which is 0.53 at confidence 0.9 and 0.83 at 0.2. With $s_{\min} = 0$ the lower
stratification bin lies entirely below that threshold for any draw, and there
the term's minimiser is an input-independent mixture — precisely the degeneracy
the term exists to prevent.

#### Where this term came from

It is not lifted from a paper. It was derived here, and the order matters: the
**negative** result came first.

**1 — Multi-step cannot come from the noise.** With a deterministic verifier, a
mean-field endpoint parameterisation and $x_0 \perp x_1 \mid \mathrm{ctx}$, the
Bayes-optimal $\pi^\theta_{0,1}$ is **constant in $x_0$**. Worse, that constant
manifold lies in the joint minimiser set of the endpoint, EC and TD terms (since removed, §8) at
$s = 0$, so no reweighting of them can exclude it. This was measured, not only
argued: eight different prior seeds at one pass produced the *same* draft — 32
generations without a single variation. Whatever a schedule with $n > 1$ buys,
it buys with **self-generated** information: the drafter's own previous output.

**2 — So what should the map be trained to compute?** If each pass is to improve
on the last, the object to imitate is one **Jacobi sweep** of the greedy AR
chain,

```math
T(\mathrm{ctx}, d)_j \;=\; \arg\max\; p_{\mathrm{AR}}\big(\cdot \mid \mathrm{ctx},\, d_{\lt j}\big)
```

— "take the whole current draft and advance every position by one AR step, in
parallel."

**3 — What composition then buys.** If $\pi^\theta_{s,1}$ realises $T$, then
induction over positions gives $A_n \ge \min(n,\,K-1)$: strictly above $A_1$
while $A_1 < K-1$, with a deterministic verifier and **without a single new
bit**. The exact map from context to chain does not lie in the class of a
fixed-depth mean-field head; it lies in that head's *composition*.

**4 — Why the objective as it stood could never get there.** Every teacher term
aimed at $p_{\mathrm{AR}}(\cdot\mid\mathrm{corpus})$, which $x_s$ does not enter
— the objective never once asked the map to read its own input. So the second
term's target has to be conditioned on the drafter's own draft,
$p_{\mathrm{AR}}(\cdot\mid\mathrm{ctx},\ \arg\max q_{k-1})$, which is exactly
$T$ applied to the current draft and evaluated by the frozen verifier — at the
cost of one forward the decode loop already spends on verification.

**5 — And the states have to be generated.** Fitting $T$ on the states the
decoder actually visits requires producing those states, which is what the
restart $x_k$ does. Training on the model's own state distribution with an
expert relabelling it is **DAgger** (Ross, Gordon and Bagnell, 2011); the
operator being imitated is the same Jacobi sweep that consistency-style parallel
decoders target. *No literature search was run to establish exact prior art for
the combination — that check is outstanding.*

**The argument that does *not* work.** "The target depends on the state,
therefore the degenerate minimiser is gone" is **false**, and worth stating
because it is the natural thing to claim. The pointwise problem
$\min_q \ell\big(T(\arg\max q),\, q\big)$ is the same for every $x_0$, and its
infimum of 0 is attained by AR's greedy continuation, which does not read the
input at all: at unbounded capacity the constant map is not merely admissible
but **optimal**. The defensible argument is about capacity — the input
*contains a partial computation of the target*, since the target is one AR sweep
applied to the draft and the draft is the input. Under `verify_kl` the input is
empty of relevant content and reading it never pays at any capacity; here it
pays at **bounded** capacity. That is the claim to defend, not identifiability.

**What stays empirical.** The guarantee above is empty as a speed claim:
$\mathrm{TPF}_n = (\min(n, K-1)+1)/(n+1) \le 1$, exactly AR speed, so a pass
pays for itself only when $A_{n+1} - A_n > \mathrm{TPF}_n$. Everything measured
above 1 comes from how far the learned operator exceeds its own guarantee, and
nothing derives that. Neither is convergence implied — `argmax` kills the
gradient through the target and the stop-gradient kills it through the input, so
this is a DAgger step, not descent on a potential. $s_{\min}$, $r = 2$ and the
term weights were chosen, not derived.

### Decoding schedule: entries stay inside the trained range

A decode cycle of $n$ passes is a list of restart pairs $(s_k, 1)$. The
multi-step term trains entries only in $[s_{\min}, 1) = [0.5, 1)$, and the
verifier term trains $s = 0$, so every entry after the first must lie in
$[s_{\min}, 1)$:

| schedule | entries | the method, 80k | Orthrus, 80k |
|---|---|---|---|
| n4 | 0 · 0.34 · 0.67 · 0.9 | 2.974 | 2.312 |
| n4v | 0 · 0.5 · 0.7 · 0.85 | **4.189** | 2.312 |

The entry at $s = 0.34$ asks the drafter to refine a state it never saw in
training and costs −0.733 ± 0.055 accepted tokens against three passes. Orthrus
has no $s$ and returns the identical number on all 460 prompts under both
schedules — direct evidence that the method's drafter is a map in $s$, not a
repeated projection under another name. Accepted tokens averaged over the six
benchmarks; README, 80k section.

### What the theory predicts about speed

If the map realised the Jacobi sweep `T(ctx,d)_j = argmax p_AR(·|ctx, d_{\lt j})`
exactly, `n` chained passes would fix positions `1 … n`, giving
`A_n ≥ min(n, K−1)`. Substituting into

```math
\mathrm{TPF} \;=\; \frac{A_n + 1}{n + 1}
```

gives **exactly 1**. The lemma is a **floor**, not a speedup mechanism:
acceleration needs `A_n > n` strictly, and the bound only gives $A_n \ge n$.
The measurements confirm it — nothing multi-step clears 1.

---

## 4. Comparison with the paper

Orthrus (arXiv 2605.12825) describes a diffusion attention whose queries, keys
and values are trained — the released weights train the output projection as
well (§3.1) — at block size 32, two epochs over 600k examples, forward KL as
the objective. Every run here uses the bench geometry instead (§3.1), so the
absolute numbers are not comparable to the paper's; contrasts within this set
are. The size of the gap is worth naming: the paper reports an average TPF of
3.89 on Qwen3-1.7B under greedy decoding, about 6.8 accepted tokens per cycle,
where Orthrus runs at 1.22 on SmolLM2-135M and 1.60 on Qwen3-0.6B after 80k
steps. The weaker regime is the one that *favours* multi-step, since a pass
pays only when it adds more accepted tokens than the current TPF.

The gap is not the harness. The released Orthrus-Qwen3-1.7B, measured with
this code, matches Table 1 within −6.4…+2.5% on the four sets whose answers end
inside the 512-token budget (§5.4). What separates the numbers is training: a
smaller model, about 189 times fewer supervised blocks, answers not regenerated
by the target model, and no output projection in the text's version
(docs/evidence.md, entries E5 and E11).

Their Table 3 reports that multi-step refinement **drops** throughput from 6.35
to 3.53 tokens per forward — a factor of 1.8 — "confirming that single-step
projection is optimal for our approach". **Our measurements agree with that
conclusion on its own metric**: extra decode passes lower tokens per forward for
every configuration in this study, including the one that wins on acceptance.

**The detail that matters:** their multi-step variant is trained by *randomly
masking 50% of block positions*, adapted from Fast-dLLM-v2 — not on the state
sequence its own decoding visits. The conclusion is drawn from a model that was
never taught multi-step refinement.

---

## 5. Results: the method against Orthrus

### 5.1 SmolLM2-135M — 20k steps, three seeds

Intervals use the **training seed** as the unit of observation (three seeds,
`df = 2`, `t₀.₉₇₅ = 4.30`), so they describe the method, not one trained model.
At three refinement passes the method leads Orthrus by **+0.835 ± 0.085**
accepted tokens (`t = 42.3`, `p = 0.0006`; per seed 0.870 / 0.834 / 0.802).

| | Orthrus | the method |
|---|---|---|
| growth from one pass to four | +0.071 ± 0.019 | **+0.893 ± 0.085** |
| tokens per forward, 1 / 3 / 4 passes | 1.219 / 0.628 / 0.507 | **1.257** / 0.824 / 0.675 |
| between-seed σ of acceptance, 1 / 3 / 4 passes | 0.004 / 0.001 / 0.001 | 0.007 / 0.032 / 0.043 |

At one pass the method is worth +0.038 tokens per forward (+3.1%, `t = 6.9`).
Wall-clock is not settled at this size: on MPS a 135M forward is dominated by
fixed overhead.

### 5.2 Qwen3-0.6B — 10k steps, Q, K, V in both arms

| accepted tokens per cycle | 1 pass | 2 | 3 | 4 (n4) |
|---|---|---|---|---|
| Orthrus | 1.837 | 1.850 | 1.883 | 1.900 |
| the method | **2.088** | **2.772** | **3.288** | **3.557** |

Paired by prompt: **+0.243 ± 0.025** at one pass, +1.595 ± 0.059 at four. The
four-pass column uses n4, outside the trained range, and understates the
method (§3.2).

### 5.3 Qwen3-0.6B — 80k steps, Q, K, V in both arms

| accepted tokens per cycle | 1 pass | 2 | 3 | 4 (n4v) | growth |
|---|---|---|---|---|---|
| Orthrus | 2.201 | 2.215 | 2.258 | 2.312 | +0.096 ± 0.017 |
| the method | **2.576** | **3.265** | **3.740** | **4.189** | **+1.559 ± 0.066** |

Paired by prompt: **+0.360 ± 0.035** at one pass (ahead on 81% of prompts),
+0.998 ± 0.047 at two (99%), +1.423 ± 0.059 at three and +1.823 ± 0.073 at four
— the last two on all 460 prompts.

| | 1 pass | 2 | 3 | 4 (n4v) |
|---|---|---|---|---|
| tokens per forward, Orthrus | 1.601 | 1.072 | 0.814 | 0.662 |
| tokens per forward, the method | **1.788** | **1.422** | **1.185** | **1.038** |
| wall-clock speedup, Orthrus | 1.347 | 0.906 | 0.693 | 0.564 |
| wall-clock speedup, the method | **1.440** | **1.130** | **0.929** | **0.805** |

Wall-clock: T4, `float32`, batch one. Under sampling at `T = 1` (Leviathan
acceptance, lossless in distribution) the margins are +0.309 ± 0.043 at one
pass, +1.636 ± 0.069 at three and +1.796 ± 0.075 at four. With generations of up
to 512 tokens instead of 64 (180 tasks) they are +0.378 ± 0.029 at one pass and
+1.809 ± 0.065 at four; the longer budget itself adds only +0.110 ± 0.074 to
Orthrus and +0.119 ± 0.087 to the method.

Tables average over the six benchmarks; paired contrasts pool the prompts, one
seed, so the prompt is the unit. Tokens per forward is the steady-state rate.

### 5.4 Context: the released Orthrus-Qwen3-1.7B in this harness

180 tasks, generations up to 512 tokens, one pass, verified per cycle against
the authors' own code (docs/evidence.md, entry E3).

| | gsm8k | math500 | humaneval | mbpp | aime24 | aime25 |
|---|---|---|---|---|---|---|
| TPF, this harness | 4.06 | 4.41 | 2.82 | 2.76 | 3.35 | 3.45 |
| TPF, the paper's Table 1 | 4.20 | 4.71 | 2.75 | 2.76 | 4.33 | 3.89 |

On AIME every generation runs into the 512-token budget, so those two columns
are not comparable. Mean acceptance 6.07 tokens per cycle, 3.02× wall-clock on a
T4.

---

## 6. Conclusions

**Multi-step training raises the single pass.** At one pass — where every
drafter is fastest — the method leads Orthrus by +0.360 ± 0.035 accepted tokens
at 0.6B and 80k steps: 1.788 against 1.601 tokens per forward (+11.7%), 1.44×
against 1.35× in wall-clock. The margin grew with the budget (+0.243 at 10k).

**Refinement keeps paying in acceptance only for the method:** +1.559 ± 0.066
from one pass to four, against +0.096 ± 0.017 for Orthrus.

**Extra passes do not pay in throughput, for either drafter.** The break-even
`A(n+1) − A(n) > TPF(n)` fails at every transition, and the paper's Table 3
says the same for its own multi-step variant. One pass is the operating point.

**The decode schedule must stay inside the trained range** — n4v, not n4.

---

## 7. Upcoming experiments, with the mathematics

### 7.1 Q, K, V, O at 100k steps

The diffusion twins are a set $\mathcal W$ of attention projections, each
initialised at its AR weight and trained; the loss of §3 is unchanged. The new
pair differs from the 80k pair in $\mathcal W$ and in the budget only:

| | Orthrus | the method |
|---|---|---|
| $\mathcal W = \{W_Q, W_K, W_V\}$, 80k (§5.3) | `qwen06_orthrus` | `qwen06_flowdraft_multistep_qkv` |
| $\mathcal W = \{W_Q, W_K, W_V, W_O\}$, 100k | `qwen06_orthrus_qkvo` | `qwen06_flowdraft_multistep` |

**Why.** The released Orthrus trains $W_O$ (§3.1), so the baseline that matches
the release has it. **What it decides.** Whether the method's one-pass margin
survives when both arms adapt $W_O$: at 10k the method lost 0.140 ± 0.027
accepted tokens at one pass by adapting it.

### 7.2 A hybrid diffusion view: LinearOrthrus and the linearized method

The AR path is untouched, so decoding stays lossless. The diffusion view
replaces some of its own components with linear maps. For a layer $\ell$ of the
plan, with $h_\ell$ the residual stream entering it:

```math
\begin{aligned}
\text{attention:}\quad & a_\ell = W^{a}_\ell\,x_\ell + b^{a}_\ell,
  && x_\ell = \mathrm{RMSNorm}^{\mathrm{in}}_\ell(h_\ell),\\
\text{MLP:}\quad & m_\ell = W^{m}_\ell\,y_\ell + b^{m}_\ell,
  && y_\ell = \mathrm{RMSNorm}^{\mathrm{post}}_\ell(h_\ell + a_\ell),\\
\text{whole layer:}\quad & h_{\ell+1} = h_\ell + W^{L}_\ell\,\mathrm{RMSNorm}^{\mathrm{in}}_\ell(h_\ell) + b^{L}_\ell .
\end{aligned}
```

Each map is distilled towards what the frozen component computes on the
drafter's own input. Where attention is replaced, that layer's twins stay at
their AR copies, so its teacher is exactly the AR attention, read through the
shared cache and the block:

```math
T^{a}_\ell(x) = W^{\mathrm{AR}}_{O,\ell}\,
\mathrm{softmax}\!\Big(\frac{Q_{\mathrm{ar}}K_{\mathrm{ar}}^{\top}}{\sqrt{d_h}}\Big)V_{\mathrm{ar}},
\qquad
T^{m}_\ell(y) = \mathrm{MLP}_\ell(y),
\qquad
T^{L}_\ell(h) = \mathrm{Layer}^{\mathrm{AR}}_\ell(h) - h .
```

```math
\mathcal L \;=\; \mathcal L_{\text{base}}
\;+\; \lambda_d\,\frac{1}{|\mathcal C|}\sum_{(\ell,c)\in\mathcal C}
\frac{\big\|\,W^{c}_\ell\,\mathrm{sg}(x) + b^{c}_\ell - \mathrm{sg}\,T^{c}_\ell(x)\big\|^2}
     {\big\|\,\mathrm{sg}\,T^{c}_\ell(x)\big\|^2}
```

$\mathcal L_{\text{base}}$ is the Orthrus KL (§3.1) or the method's loss
(§3.2); $\mathcal C$ runs over the replaced components in every diffusion pass
of the step; $\lambda_d = 1$. The input is detached inside the distillation
term, so the term trains $W$ and $b$ only, and the teacher runs under
`no_grad` — no activations are kept for it. $\mathcal L_{\text{base}}$ flows
through $Wx + b$ with $x$ attached, so the rest of the drafter adapts to the
hybrid. The maps start at zero. At decode the teacher is not computed at all: a
replaced component is skipped, which is where the speed comes from.

**The bar to clear.** At batch one a pass reads every weight once. With
$P_{\mathrm{attn}} = 2 d\, n_h d_h + 2 d\, n_{kv} d_h$,
$P_{\mathrm{mlp}} = 3 d\, d_{\mathrm{ff}}$, $P_W = d^2 + d$ and
$P_{\mathrm{total}} = L\,(P_{\mathrm{attn}} + P_{\mathrm{mlp}}) + V d$ — the output
layer is read in full — the drafting pass reads a fraction

```math
c \;=\; \frac{P_{\mathrm{total}} - \sum_{(\ell,c)\in\text{plan}} \big(P_c - P_W\big)}{P_{\mathrm{total}}}
\qquad\text{and}\qquad
S \;=\; \frac{(A_{\mathrm{hyb}} + 1)/(1 + c)}{(A_{\mathrm{full}} + 1)/2}
```

is its speed against the full drafter of the same family. On Qwen3-0.6B
attention is 30% of the read, the MLP 44% and the output layer 26% — which is
why replacing attention alone can never pay there. The plan
(`qwen06_linear_base`) has 18 components: attention in layers 2, 3, 4, 10, 11,
13, 15, 16, 19; the MLP in 0, 5, 20, 27; both in 7 and 9; the whole layer in 1,
6, 12. It gives $c = 0.745$: break-even needs
$(A_{\mathrm{hyb}} + 1) \ge 0.873\,(A_{\mathrm{full}} + 1)$, and +10% needs
$\ge 0.960$.

**How the plan was chosen.** Without training, on the 80k checkpoints of both
drafters: every component was replaced by a ridge fit in closed form,
$[W\ b] = (X^{\top}X + \lambda I)^{-1}X^{\top}Y$, and scored by the acceptance it
cost per weight saved; the plan takes the components cheapest on average over
both drafters until $c = 0.745$. Training-free, mixed plans at a similar $c$
kept 69–76% of $A + 1$ — the training has to recover the rest (docs/comparisons.md, entry C12).

Pairs: `qwen06_orthrus_qkvo_linear` against `qwen06_orthrus_qkvo`, and
`qwen06_flowdraft_multistep_linear` against `qwen06_flowdraft_multistep` —
with §7.1, a 2×2 of {Orthrus, the method} × {full, hybrid view}.

### 7.3 Two more seeds at 0.6B

Every 0.6B interval uses the prompt as the unit. Two more seeds for the Q, K, V
pair, 10k steps each, give a between-seed interval with `df = 2`, as at 135M.

### 7.4 The break-even at the paper's scale

A pass $n+1$ pays only if $A(n+1) - A(n) > \mathrm{TPF}(n)$ (§3.2). This is
measured on the released Orthrus-Qwen3-1.7B, 4B and 8B, inference only. Their
multi-pass drafting is the masked one: after pass $k$ of $n$ the
$\max\!\big(1, \mathrm{round}((K-1)(k+1)/n)\big)$ most confident positions are
committed and the rest re-masked. At $\mathrm{TPF}(1) \approx 3.5$ on the 1.7B
model (§5.4), a second pass must add more than 3.5 accepted tokens.

### 7.5 A reduced-vocabulary drafting head

Every drafting pass reads the output layer in full: $V d$ weights, 26% of a
Qwen3-0.6B pass and 18% at 1.7B. Drafting over the $V'$ most frequent tokens
lowers $c$ by $(V - V')\,d / P_{\mathrm{total}}$ — by 0.205 at $V' = 32{,}768$ on
0.6B. Decoding stays lossless, since verification uses the full vocabulary;
acceptance loses exactly the positions whose verified token lies outside $V'$.
That fraction is measured first, before anything is trained.

### 7.6 On-policy survival weights

The position weights $u_j = \partial\mathbb E[A]/\partial a_j = S_{j-1}(1 + R_j)$
(§2) are computed from a static profile, $a_j = 0.8$. Replacing it with the
drafter's own measured per-position acceptance, detached, makes the weight the
probability of reaching position $j$ under the current model.

### 7.7 Calibration for sampled decoding

Under speculative sampling a drafted token is accepted with probability
$\sum_v \min\big(q_j(v), p_j(v)\big) = 1 - \mathrm{TV}(q_j, p_j)$. Greedy
acceptance reads only the argmax, so training can push $q_j$ towards one-hot at
no cost there — and that is exactly what lowers acceptance under sampling.
Measure $\mathrm{TV}(q_j, p_j)$ on the finished checkpoints; no training, no
quota.

---

## 8. Rejected ideas

Measured and dropped. Their full sections moved to
`bucket/EXPERIMENTS_rejected_sections.md`; the code and configurations of the
rejected terms are in `bucket/`.

| idea | result | used instead |
|---|---|---|
| Masked drafter trained on its own refinement (`*_orthrus_multistep`) | 135M: +0.223 ± 0.021 over Orthrus, but only +0.046 against a control matched on projections, position weights and tail weight. 0.6B, 10k: grows from one pass to four by +0.071 ± 0.013 — no faster than Orthrus (+0.062 ± 0.010) | the same term on a continuous state — the method |
| Continuous state without the multi-step term (`*_flowdraft`, the ablation) | 135M: loses 0.563 ± 0.526 going from one pass to four; 0.6B, 10k: 1.705 at one pass against Orthrus's 1.837 | the method; at 135M the multi-step term is worth +1.138 ± 0.109 at three passes |
| Flow-consistency terms: endpoint, EC, TD | −0.114 [−0.150, −0.078] at 135M | the two-term loss of §3.2 |
| Multiplicative time conditioning | +0.027 [−0.006, +0.060], p = 0.23 | additive conditioning |
| Freezing the value projection | −0.244 [−0.281, −0.207] | training $W_V$ |
| Equal position weights | continuous −0.043 [−0.074, −0.012]; masked −0.005 | the weights $\partial\mathbb E[A]/\partial a_j$ of §2 |
| Idempotence term (`flowdraft_idem`) | at 10k, inside the trained range, +0.007 ± 0.025 at one pass; at 50k the same decode acceptance as without it. Stopped at step 50,651 | decode entries kept inside the trained range (§3.2) |
| Taylor expansion of the drafter's attention, first and second order | 83–97% of the attention mass sits on keys with $\lvert\delta\rvert > 1.5$; with every layer replaced acceptance falls to about 0 | linear maps fitted to the attention output (§7.2) |
| Linearizing the drafter without training: attention, MLP, whole layers, mixed | never faster than the full drafter; the best is parity | the trained hybrid view of §7.2 |

---

## 9. What remains open, and what the objective assumes

### Open

- **Seeds.** Three at 135M give `df = 2`; at 0.6B there is one per arm, so the
  intervals use the prompt as the unit. Two more are queued (§7.3).
- **`verify_kl` was never ablated.** At one refinement pass it does all the
  work; whether it is still needed once the multi-step term covers the restarts
  has not been measured.
- **Absolute numbers** sit far below the paper's. The reasons are in §4: the
  training, not the harness.

### Assumptions the code cannot remove

| assumption | status |
|---|---|
| Weighting `KL_j` by `∂E[A]/∂a_j` presumes the coupling `∂a_j/∂θ ≈ −κ·∂KL_j/∂θ` has a **position-independent** `κ`. Deep positions have higher target entropy, so it does not. | never measured; measurable from the per-position acceptance every run logs |
| The weight profile is **frozen** at one acceptance regime while `∂E/∂a` depends on the current point, so the stationary point is that of a linear functional `Σ c_j a_j`, not of `E`. | second-order near the anchor; at profile 0.93 against a working point near 0.6 the deep positions get ~16× more gradient mass than their contribution warrants |
| The chain-validity gate is exact **pointwise at a fixed context**. At decode the whole prefix is self-generated, not corpus text; nothing in the objective addresses that shift. | proven pointwise, including at the break position; the distribution shift is untreated |
| The prefix-fixing lemma is proven for a **token** operator. The map reads `(1−s)x₀′ + s·q`, so "realises `T` exactly" must hold for every prior draw and every `s`. | `s_min = 0.5` removes the region where recovery is provably impossible; it does not establish recovery elsewhere |
| The multi-step term is **DAgger**: `argmax` kills the target gradient, the detach kills the input gradient, and the input distribution's dependence on `θ` is discarded. The sum is therefore not the gradient of any scalar function of `θ`. | no potential, hence no convergence guarantee. Convergence was observed on every curve across three seeds; it is not implied by anything |
| Relieving the input requires a **saturating region** reachable by additive conditioning: `verify_kl` at `s = 0` wants `∂π/∂x = 0`, the multi-step term wants the opposite. | **measured and unsatisfied** — response is damped only at amplitudes 81–325× the median embedding norm, far outside what training reaches. The multiplicative gate built to relieve it gave +0.027 (p = 0.23) |

Two of these are measurable rather than permanent: the coupling constant `κ_j`
and the input Jacobian at a trained checkpoint. Neither was measured.
