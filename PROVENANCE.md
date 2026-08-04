# FlowDraft — провенанс кода и экспериментов

Рабочая записка к разделу Contribution статьи. Фиксирует, кто что создал, с
датами и способом проверки. Составлена по истории репозитория и логам wandb на
4 августа 2026. Все утверждения проверяются командами из последнего раздела.

## Зачем это нужно

В коммитах `51a30d3` (19 июля) и `d70265d` (20 июля) часть файлов была
переименована. Переименования по смыслу верные — они разводят Orthrus и CFM, —
но у них есть побочный эффект: **`git log <новое имя>` без флага `--follow`
показывает автором переименовавшего, а не создателя файла.** Любая механическая
оценка вклада по истории файлов даёт неверную картину.

## Переименования

| Было | Стало | Кто | Когда |
|---|---|---|---|
| `src/models/lit_orthrus.py` | `src/models/flowdraft.py` | V. Tekaev | 19–20 июля |
| `src/models/lit_orthrus_block_wise.py` | `src/models/flowdraft_block_wise.py` | V. Tekaev | 19–20 июля |
| `src/models/lit_orthrus_baseline.py` | `src/models/orthrus.py` | V. Tekaev | 19 июля |
| `src/models/lit_orthrus_baseline_block_wise.py` | `src/models/orthrus_block_wise.py` | V. Tekaev | 19–20 июля |
| `src/configs/experiment/baseline.yaml` | `.../orthrus.yaml` | V. Tekaev | 19 июля |
| `src/configs/experiment/baseline_block_wise.yaml` | `.../orthrus_block_wise.yaml` | V. Tekaev | 19–20 июля |
| `src/configs/experiment/flowmap_block_wise.yaml` | `.../flowdraft_block_wise.yaml` | V. Tekaev | 19–20 июля |
| `src/configs/experiment/flowmap_staged.yaml` | `.../flowdraft_staged.yaml` | V. Tekaev | 19–20 июля |

Последние две пары git не распознал как переименование (файл удалён и создан
заново), но преемственность видна по содержимому. `flowdraft_block_wise.yaml` —
дословная копия `flowmap_block_wise.yaml` с переименованными идентификаторами;
исходный комментарий автора файла сохранён без правок:

> `# ADDITION (beyond the task): the flow-map drafter retrained in the exact`
> `# inference geometry — clean AR prefix in the KV cache, clean in-block`
> `# anchor, noisy K-block.`

Текст написан 8 июля, до появления второго разработчика в репозитории (первый
коммит V. Tekaev — 12 июля). Это единственное место в проекте, где блочная
постановка помечена как расширение сверх задания.

## Создание файлов, с учётом переименований

Проверяется командой `git log --all --follow --diff-filter=A -- <файл>`.

### Модели — все созданы Ya. Sergaev

| Файл | Дата |
|---|---|
| `src/models/flowdraft.py` | 4 июля |
| `src/models/flowdraft_block_wise.py` | 4 июля |
| `src/models/orthrus.py` | 4 июля |
| `src/models/base/df_adapter.py` | 4 июля |
| `src/models/base/fte.py` | 4 июля |
| `src/models/model.py` | 4 июля |
| `src/models/factory.py` | 5 июля |

### Инфраструктура — Ya. Sergaev

`src/train.py`, `src/eval.py`, `src/data/dataloaders.py`,
`src/preprocessor/df_processor.py`, `main.py`, `src/configs/train.yaml`,
`src/configs/eval.yaml` — 4 июля; `src/plots.py` — 8 июля.

### Экспериментальные конфиги

| Конфиг | Создал | Дата |
|---|---|---|
| `orthrus.yaml` (быв. `baseline.yaml`) | Ya. Sergaev | 8 июля |
| `orthrus_block_wise.yaml` (быв. `baseline_block_wise.yaml`) | Ya. Sergaev | 8 июля |
| `flowdraft_staged.yaml` (быв. `flowmap_staged.yaml`) | Ya. Sergaev | 8 июля |
| `flowdraft_block_wise.yaml` (быв. `flowmap_block_wise.yaml`) | Ya. Sergaev | 8 июля |
| `ablate_teacher_only.yaml` | Ya. Sergaev | 8 июля |
| `ablate_consistency_only.yaml` | Ya. Sergaev | 8 июля |
| `flowdraft_packed_blockwise.yaml` | V. Tekaev | 20 июля |
| `flowdraft_packed_full.yaml` | V. Tekaev | 20 июля |
| `benchmark/orthrus.yaml`, `data/{aime24,aime25,gsm8k,humaneval,mbpp}.yaml` | V. Tekaev | 19 июля |
| `model/qwen3_1.7b.yaml` | V. Tekaev | 12 июля |

К 8 июля существовала полная экспериментальная матрица: бейзлайн, бейзлайн в
блочной геометрии, staged flow map, flow map в блочной геометрии и две абляции
целевой функции.

## Авторство живых строк (`git blame` на HEAD)

| Файл | Ya. Sergaev | V. Tekaev |
|---|---:|---:|
| `src/models/flowdraft.py` | 604 | 492 |
| `src/models/flowdraft_block_wise.py` | 105 | 384 |
| `src/models/orthrus.py` | 25 | 282 |
| `src/models/base/df_adapter.py` | 145 | 343 |
| `src/models/base/fte.py` | 40 | 0 |
| `src/models/factory.py` | 12 | 192 |
| `src/eval.py` | 203 | 131 |
| `src/train.py` | 49 | 234 |
| `src/data/dataloaders.py` | 134 | 219 |
| `src/preprocessor/df_processor.py` | 71 | 0 |
| `src/plots.py` | 108 | 0 |
| `src/models/model.py` | 52 | 25 |
| `main.py` | 67 | 10 |

Суммарно по всей истории, без merge-коммитов:

| Автор | Добавлено | Удалено |
|---|---:|---:|
| Ya. Sergaev | 7 589 | 766 |
| V. Tekaev | 6 631 | 2 896 |
| N. Nikonov | 1 528 | 2 |

## Хронология

| Дата | Событие |
|---|---|
| 3 июля | Создание репозитория (Ya. Sergaev) |
| 4 июля | Адаптер Orthrus, FlowTime-эмбеддер, обе вариации модели, спекулятивный декодинг, eval |
| 8 июля | Полная экспериментальная матрица: 6 конфигов, включая обе абляции и обе блочные постановки |
| 9 июля | Прогон `flowmap-staged`, 1329 шагов. Зафиксирован отказ: `val/acceptance_decode → 0`, `val/tpf → 0.484` (аналитический пол при нулевой приёмке) при растущем `val/teacher_agreement`. Гипотеза о несовместимости AR-дистилляции и самодистилляции опубликована в командном чате |
| 10 июля | Последний коммит Ya. Sergaev перед отпуском |
| 12 июля | Первый коммит V. Tekaev |
| 19 июля | `Make CFM logic follow the paper`: AR-KL на диагонали заменён на categorical VFM endpoint по x1, AR-член вынесен в опциональный ключ с дефолтом 0 |
| 20 июля | Packed-обучение, gradient checkpointing, переименования, кампанийные конфиги |
| 25 июля | `Add inference-aligned FlowDraft training changes`: `verify_kl = KL(sg(p_AR) ‖ π_{0,1}(x_0))`, мульти-якорные блоки |
| 26 июля | Валидационная обвязка Orthrus против авторских весов (N. Nikonov) |
| 27 июля | Последний прогон в wandb (`b8-vkl-a32-s42-screen`, 11 037 шагов) |
| 31 июля | `verify_s_uniform`: выравнивание верификатора на всём семействе последнего прыжка |

## Незакрытые пункты брифа

- **jump-count ablation** (слайд 8) — ни одного прогона с `jumps > 1`.
- **Валидация воспроизведения Orthrus** — харнесс есть, результатов в логах нет.
- **Абляции `verify_kl_weight=0` и `lambda=0`** при равном бюджете — не прогнаны.
- **Размер выборки при замере** — decode-метрики считаются по
  `val_decode_prompts: 16`, при том что разброс трёх одинаковых прогонов
  бейзлайна по TPF составляет ±9%, а заявленный эффект +3.9%.

## Как проверить любое утверждение

```bash
# создатель файла с учётом переименований
git log --all --follow --diff-filter=A --format='%an %ad' --date=short -- src/models/flowdraft.py

# все переименования и кто их сделал
git log --all --diff-filter=R --name-status -M --format='%h %an %ad'

# авторство живых строк
git blame --line-porcelain HEAD -- <файл> | grep '^author ' | sort | uniq -c | sort -rn

# вклад по строкам без merge-коммитов
git log --all --format='@@%an' --numstat --no-merges
```
