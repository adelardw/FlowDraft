# FlowDraft: Flow-Map Drafting for Lossless Parallel Decoding

> Raising the **acceptance ceiling** of lossless parallel decoding by upgrading the *drafter* to a **Categorical Flow Map** — faster generation, provably identical output.

**Language / Язык:** [English](#english) · [Русский](#русский)

<!-- Badges — TODO: fill in once the repo is public
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-WIP-orange)
-->

> 🚧 **Status: work in progress.** Method, code, and experimental results are still being added. Sections marked **TODO** are placeholders / templates.

**Summer of Machine Learning at Skoltech (SMILES) · Applied AI Center**

---

## English

### Table of contents

- [Overview](#overview)
- [Background: the decoding bottleneck](#background-the-decoding-bottleneck)
- [Host framework: Orthrus](#host-framework-orthrus)
- [The problem](#the-problem)
- [Key idea: a Categorical Flow Map drafter](#key-idea-a-categorical-flow-map-drafter)
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

Autoregressive (AR) LLMs decode strictly sequentially: generating *L* tokens costs *L* forward passes, which is memory-bandwidth bound. Diffusion LMs can draft whole blocks in parallel, but they drift from the AR distribution and lose quality. Speculative-style verification restores quality: draft in parallel, then verify against the AR model and keep only the tokens the AR model would have produced — this is **lossless**.

**FlowDraft** upgrades the *drafter* inside a lossless parallel-decoding loop. The throughput of any verify-based system is governed by its **acceptance length** — the number of drafted tokens accepted per cycle. We replace the single-step masked-diffusion drafter with a **Categorical Flow Map** drafter that produces a higher-fidelity *joint* proposal over the block at the **same** number of forward passes. Verification is left untouched, so the output stays strictly lossless — the drafter affects only **speed**, never **quality**.

### Background: the decoding bottleneck

- **AR LLMs** decode strictly sequentially: *L* tokens → *L* forward passes (memory-bandwidth bound).
- **Diffusion LMs** draft blocks in parallel, but drift from the AR distribution and lose quality.
- **Speculative-style verification** fixes quality: draft in parallel, then *verify* against the AR model → keep only correct tokens (**lossless**).

### Host framework: Orthrus

FlowDraft is built inside **Orthrus**, a lossless parallel-decoding scaffold:

- Frozen AR backbone + lightweight trainable diffusion drafter, sharing one KV cache.
- The drafter proposes *K* tokens in parallel; the frozen AR head verifies → output **provably identical** to the base model.
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

> 🚧 **TODO.** Detailed method write-up goes here: flow-map drafter architecture, the simplex endpoint head, the dual-distillation objective (AR-teacher distribution loss + flow-map consistency loss), the sampling / jump schedule, and integration with the Orthrus verification loop.

<!--
Suggested subsections to fill in:
- Notation & problem setup
- Flow-map drafter (simplex endpoint head)
- Dual distillation objective
    - AR-teacher distribution term
    - Flow-map consistency term
- Lossless verification loop (unchanged from Orthrus)
- Complexity / passes-per-block analysis
-->

### Repository structure

> 🚧 **Suggested layout (TODO: adjust as the code lands).**

```text
FlowDraft/
├── README.md
├── requirements.txt          # or environment.yml / pyproject.toml
├── configs/                  # training & experiment configs
├── src/
│   ├── orthrus/              # Orthrus reproduction (frozen AR + masked-diffusion drafter)
│   ├── flowmap/              # Categorical Flow Map drafter (simplex endpoint head)
│   ├── distillation/         # dual distillation: AR-teacher + flow-map consistency
│   ├── decoding/             # lossless parallel decoding loop + verification
│   └── eval/                 # acceptance length / TPF / throughput metrics
├── scripts/                  # train / evaluate entry points
├── notebooks/                # analysis & ablations
└── results/                  # logs, tables, figures
```

### Installation

> 🚧 **TODO.**

```bash
# git clone https://github.com/<org>/FlowDraft.git
# cd FlowDraft
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
```

### Usage

> 🚧 **TODO.** Minimal end-to-end example (load model → run lossless decoding with the flow-map drafter).

```bash
# Example (placeholder):
# python scripts/generate.py --model <path> --drafter flowmap --block-size K --jumps N
```

### Training

> 🚧 **TODO.** Dual-distillation training recipe: data, teacher, losses, schedule, hardware.

```bash
# Example (placeholder):
# python scripts/train.py --config configs/flowmap_distill.yaml
```

### Evaluation

> 🚧 **TODO.** How to reproduce the comparison and the ablations (block size, jump count), plus how losslessness is verified.

```bash
# Example (placeholder):
# python scripts/evaluate.py --config configs/eval.yaml --verify-lossless
```

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

- **Categorical Flow Maps** — Roos et al., 2026. <!-- TODO: full citation, venue, arXiv id, link -->
- **Orthrus** — lossless parallel decoding via a frozen AR backbone + trainable diffusion drafter. <!-- TODO: full citation, authors, venue, link -->

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
- [Контекст: узкое место декодирования](#контекст-узкое-место-декодирования)
- [Host-фреймворк: Orthrus](#host-фреймворк-orthrus)
- [Проблема](#проблема)
- [Идея: драфтер на Categorical Flow Map](#идея-драфтер-на-categorical-flow-map)
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

Авторегрессионные (AR) LLM декодируют строго последовательно: чтобы сгенерировать *L* токенов, нужно *L* прямых проходов — а это упирается в пропускную способность памяти. Диффузионные LM умеют драфтить целые блоки параллельно, но уходят от AR-распределения и теряют качество. Верификация в стиле спекулятивного декодирования возвращает качество: драфтим параллельно, затем сверяемся с AR-моделью и оставляем только те токены, которые выдала бы сама AR-модель — это **без потерь (lossless)**.

**FlowDraft** улучшает *драфтер* внутри lossless-петли параллельного декодирования. Пропускную способность любой системы с верификацией определяет **длина приёма (acceptance length)** — сколько сдрафченных токенов принимается за цикл. Мы заменяем одношаговый masked-diffusion драфтер на драфтер на основе **Categorical Flow Map**, который выдаёт более качественное *совместное* предложение по блоку при **том же** числе прямых проходов. Верификация не меняется, поэтому вывод остаётся строго lossless — драфтер влияет только на **скорость**, но не на **качество**.

### Контекст: узкое место декодирования

- **AR LLM** декодируют строго последовательно: *L* токенов → *L* прямых проходов (упор в пропускную способность памяти).
- **Диффузионные LM** драфтят блоки параллельно, но дрейфуют от AR-распределения и теряют качество.
- **Верификация в стиле спекулятивного декодирования** чинит качество: драфтим параллельно, затем *сверяемся* с AR-моделью → оставляем только корректные токены (**lossless**).

### Host-фреймворк: Orthrus

FlowDraft строится внутри **Orthrus** — каркаса lossless-параллельного декодирования:

- Замороженный AR-бэкбон + лёгкий обучаемый диффузионный драфтер, использующие общий KV-кэш.
- Драфтер предлагает *K* токенов параллельно; замороженная AR-голова верифицирует → вывод **доказуемо идентичен** базовой модели.
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
- **Новизна:** flow-map драфтер + **двойная дистилляция**, подгоняющая одновременно распределение AR-учителя **и** консистентность flow-map.

**Почему это важно**

1. **Эффективность** — больше длина приёма = выше пропускная способность, бесплатно.
2. **Точность (fidelity)** — ускорение при **нулевой** потере качества (это гарантирует верификация).
3. **Фундамент** — связывает дистилляцию flow-map с быстрым и точным инференсом LLM.

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

> 🚧 **TODO.** Здесь будет подробное описание метода: архитектура flow-map драфтера, симплексная голова конечной точки, объектив двойной дистилляции (loss по распределению AR-учителя + loss консистентности flow-map), расписание сэмплирования / скачков и интеграция с петлёй верификации Orthrus.

<!--
Предлагаемые подразделы для заполнения:
- Обозначения и постановка задачи
- Flow-map драфтер (симплексная голова конечной точки)
- Объектив двойной дистилляции
    - Член по распределению AR-учителя
    - Член консистентности flow-map
- Lossless-петля верификации (без изменений относительно Orthrus)
- Анализ сложности / числа проходов на блок
-->

### Структура репозитория

> 🚧 **Предлагаемая структура (TODO: скорректировать по мере появления кода).**

```text
FlowDraft/
├── README.md
├── requirements.txt          # или environment.yml / pyproject.toml
├── configs/                  # конфиги обучения и экспериментов
├── src/
│   ├── orthrus/              # воспроизведение Orthrus (замороженный AR + masked-diffusion драфтер)
│   ├── flowmap/              # драфтер Categorical Flow Map (симплексная голова)
│   ├── distillation/         # двойная дистилляция: AR-учитель + консистентность flow-map
│   ├── decoding/             # lossless-петля параллельного декодирования + верификация
│   └── eval/                 # метрики: длина приёма / TPF / пропускная способность
├── scripts/                  # точки входа train / evaluate
├── notebooks/                # анализ и абляции
└── results/                  # логи, таблицы, графики
```

### Установка

> 🚧 **TODO.**

```bash
# git clone https://github.com/<org>/FlowDraft.git
# cd FlowDraft
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
```

### Использование

> 🚧 **TODO.** Минимальный сквозной пример (загрузить модель → запустить lossless-декодирование с flow-map драфтером).

```bash
# Пример (заглушка):
# python scripts/generate.py --model <path> --drafter flowmap --block-size K --jumps N
```

### Обучение

> 🚧 **TODO.** Рецепт обучения с двойной дистилляцией: данные, учитель, лоссы, расписание, железо.

```bash
# Пример (заглушка):
# python scripts/train.py --config configs/flowmap_distill.yaml
```

### Оценка

> 🚧 **TODO.** Как воспроизвести сравнение и абляции (размер блока, число скачков) и как проверяется lossless.

```bash
# Пример (заглушка):
# python scripts/evaluate.py --config configs/eval.yaml --verify-lossless
```

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

- **Categorical Flow Maps** — Roos et al., 2026. <!-- TODO: полная ссылка, площадка, arXiv id, линк -->
- **Orthrus** — lossless-параллельное декодирование через замороженный AR-бэкбон + обучаемый диффузионный драфтер. <!-- TODO: полная ссылка, авторы, площадка, линк -->

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
