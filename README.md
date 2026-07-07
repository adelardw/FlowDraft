# FlowDraft: Flow-Map Drafting for Lossless Parallel Decoding

> Raising the **acceptance ceiling** of lossless parallel decoding by upgrading the *drafter* to a **Categorical Flow Map** — faster generation, provably identical output.

**Language / Язык:** [English](#english) · [Русский](#русский)

<!-- Badges — TODO: fill in once the repo is public
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-WIP-orange)
-->

> 🚧 **Status: core implementation landed and smoke-verified** — adapter, three training variants (fixed / block-wise / baseline; all train end-to-end on real data: SmolLM2-135M + Nemotron), lossless decoding (greedy bitwise + speculative sampling, `jumps+1` forwards per cycle), evaluation harness (mean±std, held-out val slice). GPU experiments pending; **Results** are TBD.

**Summer of Machine Learning at Skoltech (SMILES) · Applied AI Center**

---

## English

### Table of contents

- [Overview](#overview)
- [Quickstart](#quickstart)
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

# 2. Inference sanity — the UNTRAINED drafter is already lossless (just slow)
./hf-auth.sh uv run python main.py -p "Once upon a time"
#    -> generation + [lossless vs greedy AR: PASS]

# 3. Train the drafter (block_wise = the inference geometry; GPU recommended)
./hf-auth.sh uv run python src/train.py train.variant=block_wise \
    trainer.max_steps=10000 data.batch_size=8
#    watch: loss/anchor ↓, loss/ec ↓, loss/td sane, val/teacher_agreement ↑
#    checkpoints (DF head only, ~1.7 GB for 3B) land in checkpoints/

# 4. Measure acceptance / TPF vs the AR baseline (lossless asserted bitwise)
./hf-auth.sh uv run python src/eval.py checkpoint=checkpoints/last.ckpt variant=block_wise

# 5. Generate with the trained drafter (greedy; add --temperature for sampling)
./hf-auth.sh uv run python main.py -p "..." --variant block_wise \
    --checkpoint checkpoints/last.ckpt --jumps 2
```

Laptop debugging: `src/train.py` and `src/eval.py` also run on a small ungated
backbone — append the hydra overrides
`model.name=HuggingFaceTB/SmolLM2-135M-Instruct model.backbone.dtype=float32
model.backbone.device_map=null`.

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
- **Training geometries** (`train.variant`): `fixed` — noise the whole sequence; `block_wise` — the exact inference geometry: clean AR prefix in the KV cache, a CLEAN in-block anchor position (the decode loop's pending token, see below) and a noisy K-token block; also shrinks every `[B,T,V]` loss tensor to `[B,K,V]`; `baseline` — Orthrus' own single-step masked-diffusion drafter (no time conditioning, barycenter as the simplex-native `[MASK]`).
- **Decoding** (`FlowMapOrthrus.generate`): draft K fresh tokens in 1–few jumps → ONE AR forward verifies the block. The previous cycle's correction/bonus token is never committed by its own pass: it rides as a clean in-block anchor and the next verify forward commits its K/V while scoring the drafts — **cycle cost = `jumps + 1` forwards** (TPF parity with the Orthrus convention). `temperature=0`: greedy verification, output **bit-identical** to `ar_generate`. `temperature>0` (+`top_k`/`top_p`): speculative sampling (accept `min(1, p/q)`, residual resampling) — lossless **in distribution** for any drafter quality.

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
    │   ├── lit_orthrus_block_wise.py  # training in the inference geometry
    │   └── lit_orthrus_baseline.py    # Orthrus masked-diffusion baseline
    ├── preprocessor/df_processor.py   # tokenization + one-hot simplex endpoints
    ├── data/                      # YOUR Dataset / collate / DataLoader (build_dataloaders seam)
    ├── configs/                   # hydra: train.yaml, eval.yaml, model/*
    ├── train.py                   # training entrypoint
    └── eval.py                    # dataset evaluation: acceptance / TPF / speedup / NLL
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
# sampling (lossless in distribution) + a trained drafter
./hf-auth.sh uv run python main.py -p "..." --temperature 0.8 --top-k 50 \
    --jumps 2 --variant block_wise --checkpoint checkpoints/last.ckpt
```

### Training

Data: [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2),
streamed (no full download), category splits interleaved, `messages` rendered
with the tokenizer's chat template (`src/data/dataloaders.py`). Batch contract:
`input_ids [B,T]` + `attention_mask [B,T]`; the `[B,T,V]` simplex is built
on-device, never in the batch.

```bash
./hf-auth.sh uv run python src/train.py                            # fixed variant
./hf-auth.sh uv run python src/train.py train.variant=block_wise   # inference geometry
./hf-auth.sh uv run python src/train.py train.variant=baseline     # Orthrus baseline
```

Knobs live in `configs/train.yaml`: `lambda` (dual-distillation balance),
`anchor_point`, `time_sampling`, `block_size`/`min_prefix` (block-wise), optimizer,
Lightning `trainer.*`. Checkpoints store only the DF head (~1.7 GB for 3B).

### Evaluation

Prefixes of validation samples are decoded twice — flow-draft vs plain AR — and
compared. Greedy losslessness is asserted **bitwise**, not assumed.

```bash
./hf-auth.sh uv run python src/eval.py checkpoint=path.ckpt variant=block_wise
# block-size / jump-count ablation grid (hydra multirun):
./hf-auth.sh uv run python src/eval.py -m decode.block_size=4,8,16 decode.jumps=1,2,4
```

Headline metrics (mean ± std over `n_prompts`): **acceptance** per cycle and
**TPF** (tokens per forward; cycle = `jumps+1`). Wall-clock tokens/s and
speedup vs AR are reported as diagnostics (hardware/kernel dependent). The
attention kernel is a config switch (`model.backbone.attn_implementation`):
`sdpa` (default; fused, supports the DF mask — verified against eager) |
`flex_attention` (compiled block masks, GPU only) | `eager` (reference).
**Continuation NLL** under the frozen teacher is computed in sampling mode
only (at greedy the output is bitwise equal to AR, so it measures nothing).
The val slice is held out from training samples; for a distribution-level
held-out bench point `data.dataset/splits` at MATH/HumanEval-style data.

### Results

> 🚧 **TODO.** Fill in once experiments are done. All rows must be verified lossless.

| Method | Acceptance length ↑ | TPF * | Throughput (tok/s) ↑ | Lossless |
| --- | --- | --- | --- | --- |
| AR baseline | — | — | — | ✅ (trivially) |
| Orthrus (masked-diffusion drafter) | TBD | TBD | TBD | ✅ |
| **FlowDraft** (flow-map drafter) | TBD | TBD | TBD | ✅ |

\* *TPF — metric definition TBD (to be fixed in the report).*

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

# 2. Санити инференса — НЕОБУЧЕННЫЙ драфтер уже lossless (просто медленный)
./hf-auth.sh uv run python main.py -p "Once upon a time"
#    -> генерация + [lossless vs greedy AR: PASS]

# 3. Обучение драфтера (block_wise = инференсная геометрия; нужен GPU)
./hf-auth.sh uv run python src/train.py train.variant=block_wise \
    trainer.max_steps=10000 data.batch_size=8
#    смотреть: loss/anchor ↓, loss/ec ↓, loss/td без пиков, val/teacher_agreement ↑
#    чекпоинты (только DF-голова, ~1.7 ГБ для 3B) падают в checkpoints/

# 4. Замер acceptance / TPF против AR-бейзлайна (lossless утверждается побитово)
./hf-auth.sh uv run python src/eval.py checkpoint=checkpoints/last.ckpt variant=block_wise

# 5. Генерация обученным драфтером (greedy; --temperature для сэмплирования)
./hf-auth.sh uv run python main.py -p "..." --variant block_wise \
    --checkpoint checkpoints/last.ckpt --jumps 2
```

Отладка на ноутбуке: `src/train.py` и `src/eval.py` работают и на маленьком
негейтированном бэкбоне — добавьте hydra-оверрайды
`model.name=HuggingFaceTB/SmolLM2-135M-Instruct model.backbone.dtype=float32
model.backbone.device_map=null`.

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
- **Геометрии обучения** (`train.variant`): `fixed` — шумится вся последовательность; `block_wise` — в точности инференсная геометрия: чистый AR-префикс в KV-кэше, ЧИСТАЯ якорная позиция блока (pending-токен decode-петли, см. ниже) и шумный K-блок; заодно сжимает тензоры лосса `[B,T,V]` → `[B,K,V]`; `baseline` — одношаговый masked-diffusion драфтер самого Orthrus (без времени, барицентр как симплекс-нативный `[MASK]`).
- **Декодирование** (`FlowMapOrthrus.generate`): драфт K свежих токенов за 1–несколько прыжков → ОДИН AR-forward верифицирует блок. Коррекция/бонус прошлого цикла не коммитится отдельным проходом: она едет чистым якорем внутри блока, и следующая верификация коммитит её K/V, одновременно оценивая драфты — **цикл = `jumps + 1` forward'ов** (паритет TPF с конвенцией Orthrus). `temperature=0`: жадная верификация, выход **побитово** равен `ar_generate`. `temperature>0` (+`top_k`/`top_p`): спекулятивное сэмплирование (приём `min(1, p/q)`, ресэмпл из остатка) — lossless **по распределению** при любом качестве драфтера.

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
    │   ├── lit_orthrus_block_wise.py  # обучение в инференсной геометрии
    │   └── lit_orthrus_baseline.py    # masked-diffusion бейзлайн Orthrus
    ├── preprocessor/df_processor.py   # токенизация + one-hot вершины симплекса
    ├── data/                      # Dataset / collate / DataLoader (шов build_dataloaders)
    ├── configs/                   # hydra: train.yaml, eval.yaml, model/*
    ├── train.py                   # точка входа обучения
    └── eval.py                    # оценка на датасете: acceptance / TPF / speedup / NLL
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
# сэмплирование (lossless по распределению) + обученный драфтер
./hf-auth.sh uv run python main.py -p "..." --temperature 0.8 --top-k 50 \
    --jumps 2 --variant block_wise --checkpoint checkpoints/last.ckpt
```

### Обучение

Данные: [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) —
стриминг (без полной закачки), интерливинг категорийных сплитов, `messages`
рендерятся chat-template'ом токенизатора (`src/data/dataloaders.py`). Контракт
батча: `input_ids [B,T]` + `attention_mask [B,T]`; симплекс `[B,T,V]` строится
на девайсе и в батч не кладётся.

```bash
./hf-auth.sh uv run python src/train.py                            # fixed
./hf-auth.sh uv run python src/train.py train.variant=block_wise   # инференсная геометрия
./hf-auth.sh uv run python src/train.py train.variant=baseline     # бейзлайн Orthrus
```

Ручки — в `configs/train.yaml`: `lambda` (баланс двойной дистилляции),
`anchor_point`, `time_sampling`, `block_size`/`min_prefix` (block-wise), оптимизатор,
Lightning `trainer.*`. Чекпоинты хранят только DF-голову (~1.7 ГБ для 3B).

### Оценка

Префиксы валидационных сэмплов декодируются дважды — flow-draft и чистый AR — и
сравниваются. Жадный lossless утверждается **побитово**, а не предполагается.

```bash
./hf-auth.sh uv run python src/eval.py checkpoint=path.ckpt variant=block_wise
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
выход побитово равен AR и NLL не измеряет ничего). Val-срез held-out по
сэмплам обучения; для held-out по распределению направьте
`data.dataset/splits` на бенч в стиле MATH/HumanEval. Сэмплирующая оценка:
`decode.temperature>0` — выходы lossless по распределению, побитовый флаг N/A.

### Результаты

> 🚧 **TODO.** Заполнить после экспериментов. Все строки должны быть проверены на lossless.

| Метод | Длина приёма ↑ | TPF * | Пропускная способность (tok/s) ↑ | Lossless |
| --- | --- | --- | --- | --- |
| AR-бейзлайн | — | — | — | ✅ (тривиально) |
| Orthrus (masked-diffusion драфтер) | TBD | TBD | TBD | ✅ |
| **FlowDraft** (flow-map драфтер) | TBD | TBD | TBD | ✅ |

\* *TPF — определение метрики TBD (зафиксировать в отчёте).*

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