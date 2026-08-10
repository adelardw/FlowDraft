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

## 3. The loss, experiment by experiment

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

### 3.1 Orthrus, reproduced — `*_orthrus`

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
two clauses — causal AR context $\mathbf 1[k<L]\cdot\mathbf 1[k\le a_b-1]$ and
bidirectional-within-block $\mathbf 1[k\ge L]\cdot\mathbf 1[\lfloor q/K\rfloor=\lfloor (k-L)/K\rfloor]$ —
appear verbatim in both the sparse FlexAttention path and the dense fallback.

One deviation, deliberate and worth stating. The paper's objective sums over
$k=1,\dots,K$, supervising $K$ rows per block; the final mask row's target lies
one token *past* the clean block. We supervise $K-1$, dropping that row. The
same offset carries into decode, so **our `block_size=32` proposes 31 tokens
where the paper's $K=32$ proposes 32**. The cost is one extra draft slot at the
very deepest position, whose contribution to expected accepted length is the
product of all 31 preceding per-position acceptance rates. Measured over the
full campaign the mean accepted length is 1.63 and the single best block ever
observed reached 5, so that product is not distinguishable from zero at this $K$.
At small $K$ the same gap would matter and the comparison would need redoing.

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

### 3.4 Continuous state plus multi-step training — `*_flowdraft_multistep`

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
in §6.

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
from 3.2, where a slot is either masked or fixed and the only thing that can
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
manifold lies in the joint minimiser set of the endpoint, EC and TD terms at
$s = 0$, so no reweighting of them can exclude it. This was measured, not only
argued: eight different prior seeds at one pass produced the *same* draft — 32
generations without a single variation. Whatever a schedule with $n > 1$ buys,
it buys with **self-generated** information: the drafter's own previous output.

**2 — So what should the map be trained to compute?** If each pass is to improve
on the last, the object to imitate is one **Jacobi sweep** of the greedy AR
chain,

```math
T(\mathrm{ctx}, d)_j \;=\; \arg\max\; p_{\mathrm{AR}}\big(\cdot \mid \mathrm{ctx},\, d_{<j}\big)
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

### What the theory predicts about speed

If the map realised the Jacobi sweep `T(ctx,d)_j = argmax p_AR(·|ctx, d_{<j})`
exactly, `n` chained passes would fix positions `1 … n`, giving
`A_n ≥ min(n, K−1)`. Substituting into

```math
\mathrm{TPF} \;=\; \frac{A_n + 1}{n + 1}
```

gives **exactly 1**. The lemma is a **floor**, not a speedup mechanism:
acceleration needs `A_n > n` strictly, and the bound only gives $A_n \ge n$.
The measurements confirm it — nothing multi-step clears 1.

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

## 6. What remains open, and what the objective assumes

### Open

- **Three seeds give `df = 2`.** Every headline contrast clears the 4.30
  threshold comfortably, but the design is small; a fourth seed would halve the
  critical value.
- **Eight of ten runs were still improving at the step budget.** The horizon,
  not convergence, sets these numbers.
- **`verify_kl` was never ablated.** At one refinement pass it does all the work
  (the multi-step term adds +0.018); at three or four the roles invert. Whether
  the term is still needed once multi-step training covers the restarts has not
  been measured — only argued, and the first decode pass does start at `s = 0`
  where no draft exists yet.
- **Nothing has run on CUDA or on Qwen.** Collective operations, the rank seed
  offset and the sparse FlexAttention path are no-ops on a single MPS device.

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
