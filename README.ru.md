# FlowDraft: Flow-Map-драфтинг для lossless-параллельного декодирования

**Документация:** [English](README.md) · **Русский**

## Содержание

- [Обзор](#обзор)
- [Быстрый старт](#быстрый-старт)
- [Эксперименты (этапы задания)](#эксперименты-этапы-задания)
- [Контекст: узкое место декодирования](#контекст-узкое-место-декодирования)
- [Host-фреймворк: Orthrus](#host-фреймворк-orthrus)
- [Проблема](#проблема)
- [Идея: драфтер на Categorical Flow Map](#идея-драфтер-на-categorical-flow-map)
- [Коротко про обучение CFM](#коротко-про-обучение-cfm)
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

## Обзор

Авторегрессионные (AR) LLM декодируют строго последовательно: чтобы сгенерировать *L* токенов, нужно *L* прямых проходов — а это упирается в пропускную способность памяти. Диффузионные LM умеют драфтить целые блоки параллельно, но уходят от AR-распределения и теряют качество. Верификация в стиле спекулятивного декодирования возвращает качество: драфтим блок параллельно, затем сверяем его с AR-моделью за один проход и оставляем только те токены, которые выдала бы сама AR-модель — это **без потерь (lossless)**.

**FlowDraft** улучшает *драфтер* внутри lossless-петли параллельного декодирования. Пропускную способность любой системы с верификацией определяет **длина приёма (acceptance length)** — сколько сдрафченных токенов принимается за цикл. Мы заменяем одношаговый masked-diffusion драфтер на драфтер на основе **Categorical Flow Map**, который выдаёт более качественное *совместное* предложение по блоку при **том же** числе прямых проходов. Верификация не меняется, поэтому вывод остаётся строго lossless — драфтер влияет только на **скорость**, но не на **качество**.

Важно: верифицирует именно AR-модель, поэтому она остаётся **замороженной на всех этапах**. Именно то, что её не трогают, и делает вывод доказуемо идентичным базовой модели — на этом держится слово *lossless*.

## Быстрый старт

```bash
# 1. Установка (один раз)
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env        # доступ к gated meta-llama
./hf-auth.sh                         # проверка: печатает ваш HF-логин

# 2. Проверка, что инференс работает — НЕОБУЧЕННЫЙ драфтер уже lossless (просто медленный)
./hf-auth.sh uv run python main.py -p "Once upon a time"
#    -> генерация + [lossless vs greedy AR: PASS]

# 3. Обучение FlowDraft (полноследовательный рецепт; нужен GPU)
./hf-auth.sh uv run python src/train.py \
    trainer.max_steps=10000 data.batch_size=8
#    смотреть: loss/endpoint ↓, loss/ec ↓, loss/td без пиков, val/teacher_agreement ↑
#    чекпоинты (DF-голова + Adam-моменты, ~5 ГБ для 3B) падают в checkpoints/
#    наше ДОПОЛНЕНИЕ сверх задания — обучение в точной инференсной геометрии:
#    добавьте train.variant=flowdraft_block_wise
#    эпохи поверх стриминга (ничего не скачивается заранее): фиксированный пул
#    из N сэмплов, повторённый M раз, каждый повтор в новом порядке —
#    ./hf-auth.sh uv run python src/train.py \
#        data.train_size=471952 trainer.max_epochs=2 trainer.max_steps=7375

Для обучения на нескольких GPU используйте DDP Lightning и укажите число GPU:

```bash
./hf-auth.sh uv run python src/train.py \
    trainer.accelerator=gpu trainer.devices=2 trainer.strategy=ddp \
    data.batch_size=8 trainer.max_steps=10000
```

Во время обучения `model.backbone.device_map` всегда отключается: device map Hugging Face нужен для inference-шардинга, а DDP требует полную реплику модели на каждом GPU. `data.batch_size` в команде задан на один GPU; для большего эффективного глобального batch используйте `trainer.accumulate_grad_batches`.

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

## Эксперименты (этапы задания)

Каждый этап задания проекта — один пресет, то есть готовый набор настроек (`src/configs/experiment/`). Чекпоинты падают
в `checkpoints/` (DF-голова + Adam-моменты, без бэкбона), кривые обучения — в TensorBoard:
`uv run tensorboard --logdir checkpoints`.

Те же метрики можно отправлять в Weights & Biases. Один раз выполните
`uv run wandb login` (или задайте `WANDB_API_KEY`), затем включите sink:

```bash
./hf-auth.sh uv run python src/train.py \
    wandb.enabled=true wandb.project=flowdraft wandb.name=qwen2-flowdraft
```

`wandb.offline=true` сохраняет прогон локально для последующего `wandb sync`.

**Словарь кривых обучения:**

| Кривая | Что означает | Что ждать |
|---|---|---|
| `train/loss` | полный лосс шага | ↓ (шумно: каждый шаг — свежие `(s,t)` и точка разреза) |
| `loss/endpoint` | `CE(x1, π_{t,t}(x_t))` — категориальный VFM-лосс диагонали | монотонно ↓ |
| `loss/ar_kl` | опциональный `KL(sg(p_AR) ‖ π_{t,t})`; по умолчанию вес 0 | смотреть только при `ar_kl_weight>0` |
| `loss/ec` | согласие прыжка с диагональю в точке приземления | ↓ по мере обучения диагонали |
| `loss/td` | временной дрейф `‖∂_t π‖²` | всплеск после старта, затем умеренные значения; устойчивый 0 у обученной модели = мёртвый time-канал (сигнал к adaLN-апгрейду) |
| `loss/lambda` | текущий вес ECLD | рамп 0→λ при staged, константа иначе |
| `val/teacher_agreement` | доля позиций, где argmax драфтера = argmax верификатора; прокси acceptance | **главная кривая**: ↑ = проект едет |

**Этап 1 — бейзлайн Orthrus:**

```bash
./hf-auth.sh uv run python src/train.py +experiment=orthrus
```

Одношаговая masked-diffusion голова самого Orthrus (барицентр как `[MASK]`)
на 0.5B бэкбоне, полноследовательный рецепт: случайная доля позиций
маскируется, голова восстанавливает их под один KL-член к замороженному
учителю. Смотреть: `train/loss` ↓, `val/teacher_agreement` ↑ (бейзлайн
стартует выше flow-вариантов — реконструкция при видимых соседях проще
транспорта из шума).

**Этапы 2–3 — flow-map драфтер + staged VFM/ECLD:**

```bash
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_staged
```

CFM-драфтер в той же геометрии; λ рампится 0→1 за 2000 шагов — сначала
endpoint-инференс диагонали, затем консистентность. Смотреть:
`loss/endpoint` падает с первого шага; `loss/ec` включается по мере рампа
(виден на `loss/lambda`); `val/teacher_agreement` растёт медленнее
бейзлайна на старте — норма, драфт из чистого шума требует больше шагов.
Вариант со статичным λ (без staging):
`./hf-auth.sh uv run python src/train.py model=qwen3_1.7b`.

**Этап 4 — lossless при сэмплировании:**

```bash
# coupled (дефолт): побитовый lossless при T>0 — тот же сид даёт тот же текст, что AR
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b checkpoint=<ckpt> decode.temperature=0.8
# uncoupled (Левиафан): эквивалентность законов, нуль-калиброванный TV-тест
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b checkpoint=<ckpt> \
    decode.temperature=0.8 decode.coupled=false decode.equiv_samples=500
```

**Этап 5 (абляции) — вклад каждого члена дистилляции:**

```bash
./hf-auth.sh uv run python src/train.py +experiment=ablate_teacher_only        # только endpoint (lambda=0; legacy-имя)
./hf-auth.sh uv run python src/train.py +experiment=ablate_consistency_only    # только консистентность (endpoint_weight=0)
```

Ожидания: endpoint-only — диагональ учится, прыжки без сигнала → acceptance
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
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b variant=orthrus checkpoint=<ckpt>
# (iii) flow-map, 1 прыжок (основной замер)
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b variant=flowdraft checkpoint=<ckpt>
# (iv) flow-map, несколько прыжков: кривая acceptance от числа проходов (сетка K x jumps).
#      ВАЖНО: decode.block_size = K, число токенов черновика за цикл на ИНФЕРЕНСЕ —
#      ручка ЛЮБОГО варианта; с вариантом ОБУЧЕНИЯ flowdraft_block_wise она
#      не связана, несмотря на похожее имя
./hf-auth.sh uv run python src/eval.py -m model=qwen3_1.7b variant=flowdraft checkpoint=<ckpt> \
    decode.block_size=4,8,16 decode.jumps=1,2,4
# каждая строка выше — на MATH-500, которого не было в обучении (дефолт data=math500);
# парная цифра на обучающем распределении — тот же прогон с data=nemotron:
./hf-auth.sh uv run python src/eval.py model=qwen3_1.7b variant=flowdraft checkpoint=<ckpt> data=nemotron
```

**Дополнение (сверх задания) — обучение в инференсной геометрии:**

Этапы задания выше используют полноследовательные варианты (`flowdraft` /
`orthrus`) — в самом задании про геометрию обучения ничего не сказано. Наше
дополнение: те же два драфтера, переобученные в точной геометрии
декодирования (block-causal: каузально к закэшированному префиксу,
двунаправленно внутри блока, чистый якорь) — варианты `flowdraft_block_wise` /
`orthrus_block_wise`. Это убирает расхождение геометрий обучения и
инференса и даёт сравнение «геометрия-в-геометрию»:

```bash
./hf-auth.sh uv run python src/train.py +experiment=orthrus_block_wise
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_block_wise
# оценка: те же команды этапа 5 с variant=orthrus_block_wise / flowdraft_block_wise
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

## Контекст: узкое место декодирования

- **AR LLM** декодируют строго последовательно: *L* токенов → *L* прямых проходов (упор в пропускную способность памяти).
- **Диффузионные LM** драфтят блоки параллельно, но дрейфуют от AR-распределения и теряют качество.
- **Верификация в стиле спекулятивного декодирования** чинит качество: драфтим параллельно, затем *сверяемся* с AR-моделью → оставляем только корректные токены (**lossless**).

## Host-фреймворк: Orthrus

FlowDraft строится внутри **Orthrus** — каркаса lossless-параллельного декодирования:

- Один трансформер, две attention-ветки: **замороженная AR-ветка** и **лёгкая обучаемая диффузионная ветка** (~16% параметров), использующие общие norm / MLP / эмбеддинги и единый KV-кэш.
- Диффузионная ветка предлагает *K* токенов параллельно; замороженная AR-голова верифицирует их за один проход → вывод **доказуемо идентичен** базовой модели. Принятые токены коммитятся в общий KV-кэш, и цикл продолжается со следующего блока.
- По данным Orthrus: ускорение до **7.8×**, обучается лишь **~16%** параметров на **<1B** токенов.

> *Эти цифры относятся к host-фреймворку Orthrus (предыдущая работа), а не к результатам самого FlowDraft.*

## Проблема

- Пропускная способность любой системы с верификацией = **длина приёма** (принятых сдрафченных токенов за цикл).
- Драфтер Orthrus — **одношаговая masked diffusion** модель → предполагает, что позиции в блоке условно независимы → драфты расходятся → токены отклоняются.
- Уточнение драфта помогло бы, но **лишний шаг стоит прямого прохода** и снижает пропускную способность.
- Нужно **лучшее предложение за проход**, а не больше проходов.

## Идея: драфтер на Categorical Flow Map

- **Categorical Flow Maps** [Roos et al., 2026] выучивают *интегрированное, скоррелированное* распределение конечной точки на симплексе и генерируют за **один или несколько скачков (jumps)**.
- Используем это как драфтер: **более качественное совместное предложение** по блоку — при **том же числе проходов**.
- Верификация не меняется → вывод остаётся **строго lossless**; драфтер влияет только на *скорость*, но не на *качество*.
- **Новизна:** flow-map драфтер внутри Orthrus с категориальным VFM endpoint-инференсом, ECLD-консистентностью и опциональным выравниванием по AR-верификатору.

**Почему это важно**

1. **Эффективность** — больше длина приёма = выше пропускная способность, бесплатно.
2. **Точность (fidelity)** — ускорение при **нулевой** потере качества (это гарантирует верификация).
3. **Фундамент** — связывает дистилляцию flow-map с быстрым и точным инференсом LLM.

## Коротко про обучение CFM

Драфтер учит две взаимодополняющие части категориальной flow map:

- **Endpoint inference — *какая конечная точка принадлежит траектории*.** Диагональный предиктор учится категориальным VFM-лоссом на чистой конечной точке, использованной в интерполянте.
- **Self-consistency — *как прыгать*.** Надёжная диагональ в перенесённой точке учит дальний прыжок через ECLD.

ECLD-цель отделена stop-gradient. AR-KL к верификатору включается отдельно через `train.ar_kl_weight`, но по умолчанию равен нулю, поскольку не входит в CFM-объектив статьи.

AR-модель остаётся замороженной: в block-wise обучении она даёт точный KV-префикс, а на инференсе проверяет каждое предложение и тем самым гарантирует lossless.

## Цели

1. **Воспроизвести Orthrus** (замороженный AR + masked-diffusion драфтер, общий KV-кэш, lossless-петля) в приемлемом масштабе.
2. **Реализовать flow-map драфтер** (симплексная голова конечной точки, 1–несколько скачков).
3. **Обучить категориальную flow map** (VFM endpoint inference + ECLD, с опциональным AR-KL).
4. **Оценить и сравнить:** AR-бейзлайн vs. masked-diffusion Orthrus vs. flow-map драфтер — по длине приёма, TPF и пропускной способности — всё с проверкой на lossless.

## Ожидаемые результаты

1. Воспроизведение lossless-параллельного декодера Orthrus (masked-diffusion драфтер).
2. Реализация **драфтера Categorical Flow Map** + обучение VFM/ECLD.
3. Оценка: сравнение по длине приёма / TPF / пропускной способности, с проверкой lossless и **абляциями по размеру блока / числу скачков**.

## Метод

Один замороженный бэкбон, два attention-пути (каркас Orthrus) и CFM-драфтер на диффузионном пути с VFM endpoint-инференсом и ECLD. Реализовано; масштабная валидация впереди.

- **Адаптер** (`src/models/base/df_adapter.py`): каждый `q/k/v_proj` получает обучаемого двойника-копию замороженного AR-веса (~14% параметров 3B). Роутинг stateless (`torch.func.functional_call`, дерево модулей бэкбона не модифицируется); norm / MLP / `o_proj` / эмбеддинги / LM-head и единый KV-кэш — общие. Кэш AR-only по контракту: драфтер читает закоммиченный префикс, его собственные K/V срезаются сразу после каждого прохода. DF-путь работает **без маски** (двунаправленно; CFM не требует маски, кроме паддинга) и кондиционируется временами прыжка `(s, t)` через синусоидальный time-эмбеддинг с нулевой инициализацией (`fte.py`).
- **Объектив** (`FlowDraft.compute_loss`): `loss = endpoint_weight·endpoint + ar_kl_weight·AR_KL + λ·(4·EC + 2·TD)`
  - **endpoint** — `CE(x1, π_{t,t}(x_t))`: категориальный VFM-лосс статьи. Paper-faithful режим — `train.anchor_point=trajectory`; `landing` оставлен как эксперимент.
  - **AR KL** — опциональный `KL(sg(p_AR) ‖ π_{t,t})`, выключен по умолчанию, поскольку не входит в CFM-объектив статьи.
  - **EC** — ур. (18) из *Categorical Flow Maps*: `CE(sg(π_{t,t}(X_{s,t}(x_s))), π_{s,t}(x_s))` — прыжки учатся у диагонали в точке собственного приземления; знание течёт `x1 → π_{t,t} → π_{s,t}`.
  - **TD** — ур. (16): временной дрейф `‖∂_t π_{s,t}‖²`.
  - Пары `(s, t)` на сэмпл (`train.time_sampling`): `paper` (дефолт: t~U, s~U[0,t]) | `triangle` | `sequential`.
- **Геометрии обучения** (`train.variant`): варианты из задания — полноследовательные: `flowdraft` (шумится вся последовательность) и `orthrus` (одношаговый masked-diffusion драфтер самого Orthrus: без времени, барицентр как симплекс-нативный `[MASK]`). Наше **дополнение сверх задания**: `flowdraft_block_wise` / `orthrus_block_wise` — те же два драфтера, переобученные в точности в инференсной геометрии (чистый AR-префикс в KV-кэше, ЧИСТАЯ якорная позиция блока — pending-токен decode-петли, см. ниже — и шумный K-блок; заодно сжимает тензоры лосса `[B,T,V]` → `[B,K,V]`).
- **Декодирование** (`FlowDraft.generate`): драфт K свежих токенов за 1–несколько прыжков → ОДИН AR-forward верифицирует блок. Коррекция/бонус прошлого цикла не коммитится отдельным проходом: она едет чистым якорем внутри блока, и следующая верификация коммитит её K/V, одновременно оценивая драфты — **цикл = `jumps + 1` forward'ов** (паритет TPF с конвенцией Orthrus). `temperature=0`: жадная верификация, выход **побитово** равен `ar_generate`. `temperature>0` с Gumbel-связыванием (дефолт): пер-позиционный Gumbel-шум делает сэмплирование детерминированным argmax'ом — выход **побитово** равен сэмплирующему `ar_generate` с тем же сидом. Без связывания (`coupled=false`): спекулятивное сэмплирование Левиафана, lossless **по распределению**.

## Структура репозитория

```text
FlowDraft/
├── main.py                        # playground-CLI (typer): генерация из ваших промптов
├── hf-auth.sh                     # HF_TOKEN из .env -> окружение (gated Llama)
├── pyproject.toml                 # uv-проект; ставится editable-пакетом `src`
└── src/
    ├── models/
    │   ├── base/df_adapter.py     # FlowDraftAttentionAdapter: замороженный AR + DF-двойники
    │   ├── base/fte.py            # FlowTimeEmbedding (s, t)
    │   ├── model.py               # build_model: бэкбон + токенизатор + процессор
    │   ├── factory.py             # build_lit: выбор варианта + загрузка чекпоинта
    │   ├── flowdraft.py           # FlowDraft: лосс, обучение, lossless generate
    │   ├── flowdraft_block_wise.py        # FlowDraft в инференсной геометрии
    │   ├── orthrus.py             # masked-драфтер Orthrus (full-sequence)
    │   └── orthrus_block_wise.py          # masked-драфтер Orthrus, block-causal
    ├── preprocessor/df_processor.py   # токенизация + one-hot вершины симплекса
    ├── data/dataloaders.py        # стриминговый Dataset / collate / DataLoader;
    │                              #   EpochShuffled: повторы в новом порядке (эпохи)
    ├── configs/                   # hydra-конфиги
    │   ├── train.yaml             # конфиг точки входа обучения
    │   ├── eval.yaml              # конфиг точки входа оценки
    │   ├── model/                 # qwen3_1.7b (дефолт) | qwen2_0.5b | llama3_3b
    │   ├── data/                  # nemotron (обучение) | math500 (оценка, не было в обучении)
    │   └── experiment/            # по пресету на этап задания + дополнения:
    │                              #   orthrus | flowdraft_staged | ablate_teacher_only |
    │                              #   ablate_consistency_only | orthrus_block_wise |
    │                              #   flowdraft_block_wise
    ├── train.py                   # точка входа обучения
    ├── eval.py                    # оценка на датасете: acceptance / TPF / NLL -> results/eval.jsonl
    └── plots.py                   # фигуры отчёта: frontier / TPF-бары / TPF-от-K
```

## Установка

```bash
git clone https://github.com/<org>/FlowDraft.git && cd FlowDraft
uv sync
echo "HF_TOKEN=hf_..." > .env     # доступ к gated meta-llama
./hf-auth.sh                      # проверить аутентификацию
```

## Использование

```bash
# генерация из ваших промптов (greedy: с побитовой lossless-проверкой)
./hf-auth.sh uv run python main.py -p "Once upon a time" -p "def main():"
# сэмплирование — тоже побитово равно AR (Gumbel-связывание — дефолт; --no-coupled = lossless по распределению)
./hf-auth.sh uv run python main.py -p "..." --temperature 0.8 --top-k 50 \
    --jumps 2 --checkpoint checkpoints/last.ckpt
```

## Обучение

Данные: [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) —
стриминг (без полной закачки), интерливинг категорийных сплитов, `messages`
рендерятся chat-template'ом токенизатора (`src/data/dataloaders.py`). Контракт
батча: `input_ids [B,T]` + `attention_mask [B,T]`; симплекс `[B,T,V]` строится
на девайсе и в батч не кладётся.

```bash
./hf-auth.sh uv run python src/train.py                            # FlowDraft (рецепт задания)
./hf-auth.sh uv run python src/train.py train.variant=orthrus     # бейзлайн Orthrus (из задания)
./hf-auth.sh uv run python src/train.py train.variant=flowdraft_block_wise   # ДОПОЛНЕНИЕ: инференсная геометрия
```

Ручки — в `configs/train.yaml`: `lambda`/`endpoint_weight`/`ar_kl_weight`/`lambda_ramp_steps`
(баланс VFM/ECLD + staging), `anchor_point`, `time_sampling`,
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
./hf-auth.sh uv run python src/train.py +experiment=orthrus \
    data.train_size=471952 data.batch_size=8 data.max_length=2048 \
    trainer.accumulate_grad_batches=16 train.lr=2e-4 \
    trainer.max_epochs=2 trainer.max_steps=7375
```

(`max_steps` = 471952 сэмплов / (8 × 16) на оптимизаторный шаг × 2 эпохи;
косинусному расписанию он нужен явно — у стримингового загрузчика нет длины.)

## Справочник конфигов

Все конфиги — в `src/configs/` (hydra). Любой ключ переопределяется из
командной строки (`train.lr=3e-4`), группы меняются целиком
(`model=qwen3_1.7b data=nemotron`), пресеты добавляются `+experiment=...`.

**`train.yaml` — обучение (`src/train.py`)**

| Ключ | Дефолт | Что делает |
| --- | --- | --- |
| `seed` | 42 | глобальный сид: перемешивание данных, шум, инициализация |
| `output_dir` | `checkpoints` | куда падают чекпоинты, логи TensorBoard и локальные данные W&B |
| `wandb.enabled` | false | отправлять все training/validation-метрики Lightning в W&B |
| `wandb.project` / `entity` / `name` | `flowdraft` / null / null | проект, команда/пользователь и имя прогона; null берёт дефолт W&B |
| `wandb.group` / `tags` | null / [] | необязательная группировка прогонов в W&B |
| `wandb.offline` | false | писать локально для последующего `wandb sync`, не загружая сразу |
| `train.variant` | `flowdraft` | какой драфтер учить: из задания — `flowdraft` \| `orthrus` (полноследовательные); дополнение — `flowdraft_block_wise` \| `orthrus_block_wise` (инференсная геометрия) |
| `train.block_size` | 64 | K — длина блока на обучении (block-wise варианты) |
| `train.min_prefix` | 1 | минимальный чистый префикс перед блоком |
| `train.lr` / `weight_decay` / `betas` | 1e-4 / 0.01 / [0.9, 0.95] | AdamW только по DF-голове; `lr` — ПИК расписания |
| `train.lr_schedule` | `cosine` | `cosine` (линейный разогрев → косинусный спад к 0; нужен конечный `trainer.max_steps` или `limit_train_batches`+`max_epochs`) \| `constant` |
| `train.warmup_ratio` | 0.05 | только cosine: доля шагов на разогрев |
| `train.time_sampling` | `paper` | как сэмплируются пары (s, t): `paper` \| `triangle` \| `sequential` |
| `train.lambda` | 1.0 | вес consistency-части (4·EC + 2·TD) |
| `train.endpoint_weight` | 1.0 | вес категориального VFM endpoint CE; 0 = абляция без endpoint-якоря |
| `train.ar_kl_weight` | 0.0 | опциональный AR-KL к верификатору; 0 сохраняет paper-faithful CFM-объектив |
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
| `variant` | `flowdraft` | должен совпадать с тем, как учили чекпоинт |
| `results_file` | `results/eval.jsonl` | каждый прогон дописывает JSON-строку (вход `src/plots.py`) |
| `decode.block_size` / `decode.jumps` | 8 / 1 | K и число уточняющих проходов на инференсе — ручки ЛЮБОГО варианта; `block_size` НЕ связан с вариантом обучения `flowdraft_block_wise` (см. гид ниже) |
| `decode.max_new_tokens` | 64 | сколько токенов генерировать на промпт |
| `decode.n_prompts` | 64 | сколько промптов взять из датасета (100–200 для таблицы) |
| `decode.prompt_len` | null | null = полный отрендеренный промпт; число N = первые N токенов |
| `decode.temperature` / `top_k` / `top_p` | 0 / null / null | 0 = greedy; >0 = сэмплирование |
| `decode.coupled` | true | T>0: Gumbel-связанное сэмплирование — побитово равно AR |
| `decode.equiv_samples` | 0 | только uncoupled: N выборок для TV-теста равенства законов; 0 = выкл |

**`model/*` — бэкбон** (`qwen3_1.7b` по умолчанию; `qwen2_0.5b` и `llama3_3b` остаются доступными):
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
| `orthrus` | `variant=orthrus` | `checkpoints/orthrus/orthrus-*.ckpt` |
| `flowdraft_staged` | `variant=flowdraft`, `lambda_ramp_steps=2000` | `checkpoints/flowdraft-staged/flowdraft-staged-*.ckpt` |
| `ablate_teacher_only` | `variant=flowdraft`, `lambda=0` (endpoint-only; legacy-имя) | `checkpoints/ablate-endpoint/ablate-endpoint-*.ckpt` |
| `ablate_consistency_only` | `variant=flowdraft`, `endpoint_weight=0` | `checkpoints/ablate-consistency/ablate-consistency-*.ckpt` |
| `orthrus_block_wise` (дополнение) | `variant=orthrus_block_wise` | `checkpoints/orthrus-block-wise/orthrus-block-wise-*.ckpt` |
| `flowdraft_block_wise` (дополнение) | `variant=flowdraft_block_wise`, `lambda_ramp_steps=2000` | `checkpoints/flowdraft-block-wise/flowdraft-block-wise-*.ckpt` |

Свой эксперимент (например, серия по `anchor_point` после основных этапов,
или block-wise против full-sequence FlowDraft) — переопределите имя и каталог, чтобы у него
тоже были свои каталог и имя:

```bash
./hf-auth.sh uv run python src/train.py +experiment=flowdraft_staged \
    train.anchor_point=landing \
    output_dir=checkpoints/anchor-landing 'train.checkpoint_name="anchor-landing-{step:07d}"'
```

## Параметры инференса простыми словами

Один цикл декодирования устроен так: драфтер угадывает сразу целый блок
токенов, замороженная базовая модель проверяет догадку за один проход,
совпавшее начало блока принимается, и базовая модель добавляет один свой
токен (исправление первой ошибки — или бонус, если совпало всё). Дальше —
следующий цикл. Ручки:

- `--block-size` (K) — сколько токенов драфтер угадывает за цикл. Больше —
  выше потенциальное ускорение, но хвост длинной догадки опирается на её же
  непроверенное начало и отбрасывается чаще. Перебирайте 4–16.
  Несмотря на похожее имя, к варианту обучения `flowdraft_block_wise` это отношения
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

## Оценка

Промпты датасета (по умолчанию полные; `decode.prompt_len=N` — первые N
токенов) декодируются дважды — flow-draft и чистый AR — и сравниваются. Жадный lossless утверждается **побитово**, а не предполагается.

```bash
./hf-auth.sh uv run python src/eval.py checkpoint=path.ckpt   # variant=flowdraft по умолчанию
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

## Результаты

> 🚧 **TODO.** Заполнить после экспериментов. Все строки должны быть проверены на lossless.

| Метод | Длина приёма ↑ | TPF * | Пропускная способность (tok/s) ↑ | Lossless |
| --- | --- | --- | --- | --- |
| AR-бейзлайн | — | — | — | ✅ (тривиально) |
| Orthrus (masked-diffusion драфтер) | TBD | TBD | TBD | ✅ |
| **FlowDraft** (flow-map драфтер) | TBD | TBD | TBD | ✅ |

\* *TPF — токенов на прямой проход: `N сгенерировано / N проходов`, цикл = `jumps + 1` проходов (формулы — в гиде выше).*

**Абляции (TODO):** размер блока, число скачков.

## Ссылки

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

## Команда

- **Участники:** <!-- TODO: члены команды -->
- **Кураторы / менторы:** Maria Ivanova (YSDA, Applied AI Institute) · Dmitrii Babaev

## Благодарности

Работа выполнена в рамках **Summer of Machine Learning at Skoltech (SMILES)**, Skoltech Applied AI Center.

## Лицензия

> 🚧 **TODO:** выбрать и добавить лицензию (например, MIT / Apache-2.0).
