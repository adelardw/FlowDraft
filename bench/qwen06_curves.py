"""Кривые стенда Qwen3-0.6B по логам, скачанным с Kaggle.

Отдельно от `curves.py`: тот читает `checkpoints/`, а здесь источник —
`kaggle_out/*/qwen06_*/lightning_logs`, и все четыре прогона УКОРОЧЕНЫ на
разном числе шагов. Поэтому сравнение делается на общем шаге, а не на конце
каждой кривой: конец у всех четырёх разный и сам по себе ничего не значит.

Рисуется только то, что логируют ВСЕ четыре. `val/accepted_tokens_per_block`
пишет лишь маскирующая ветвь, а `val/loss` у вариантов считает разные
объективы — на одной оси это была бы подмена.
"""
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
SLOT = ["#8b8a85", "#eb6834", "#eda100", "#2a78d6"]
# Пятая запись — та же конфигурация без o_proj, поэтому цвет тот же, а стиль
# другой: цвет = эксперимент, как во всех фигурах README.
RUNS = [
    ("qwen06_orthrus", "Orthrus, reproduced (Q,K,V)", SLOT[0], "-"),
    ("qwen06_orthrus_multistep", "masked + multi-step training (Q,K,V,O)", SLOT[1], "-"),
    ("qwen06_flowdraft", "continuous state, no multi-step (Q,K,V,O)", SLOT[2], "-"),
    ("qwen06_flowdraft_multistep", "continuous + multi-step (Q,K,V,O)", SLOT[3], "-"),
    ("qwen06_flowdraft_multistep_qkv", "continuous + multi-step (Q,K,V)", SLOT[3], "--"),
]
ARMS = [(k, l) for k, l, _, _ in RUNS]
COLOR = {k: c for k, _, c, _ in RUNS}
STYLE = {k: st for k, _, _, st in RUNS}
LABEL = dict(ARMS)
BLOCK = 32


def load():
    out = {}
    # Траектория конфигурации СШИТА из нескольких прогонов: обучение шло с
    # возобновлениями, и каждый сеанс Kaggle писал свой каталог. `version_*`, а
    # не `version_0`: при откате по памяти на меньший батч Lightning заводит
    # новую версию. `r1_timeout` исключён — это переучивание со старого
    # чекпоинта после того, как ядро смонтировало предыдущую версию датасета,
    # его точки дублируют настоящие.
    for ev in glob.glob("kaggle_out/*/qwen06_*/lightning_logs/version_*/events*"):
        if "r1_timeout" in ev:
            continue
        exp = ev.split("/")[2]
        ea = EventAccumulator(ev, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for sc in ea.Scalars(tag):
                out.setdefault(exp, {}).setdefault(tag, {})[sc.step] = sc.value
    return out


def series(run, tag):
    return run.get(tag, {})


def draw(ax, xs, ys, color, style, label=None, note_below=False):
    """Одна точка — это не линия, и рисовать её как линию бессмысленно.

    У версии Q,K,V промежуточные валидации потерялись вместе с каталогом
    вывода, который два её отрезка обучения делили между сессиями. Kaggle
    хранит вывод только последней версии ядра, поэтому восстановить их нечем.
    Рисуем крупным ромбом с подписью — видно, что точка есть, и видно, что
    траектории у неё нет.
    """
    if len(xs) == 1:
        ax.plot(xs, ys, color=color, marker="D", ms=11, ls="none", label=label,
                markeredgecolor="#fcfcfb", markeredgewidth=1.4, zorder=5)
        ax.annotate("single point:\nearlier validations lost", (xs[0], ys[0]),
                    color=color, fontsize=7.5, ha="right", va="top" if note_below else "center",
                    xytext=(-14, -12) if note_below else (-14, 0), textcoords="offset points")
    else:
        ax.plot(xs, ys, color=color, ls=style, lw=2, marker="o", ms=5, label=label)


def main():
    E = load()
    missing = [a for a, _ in ARMS if a not in E]
    if missing:
        print(f"нет логов: {missing}", file=sys.stderr)
    runs = [a for a, _ in ARMS if a in E]
    # Общий шаг — последний, на котором валидировались ВСЕ. Иначе панель
    # сравнивала бы модель на 4428 шагах с моделью на 2806.
    steps_each = [set(series(E[a], "val/teacher_agreement")) for a in runs]
    common = sorted(set.intersection(*steps_each)) if steps_each else []
    S = common[-1] if common else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)

    for a in runs:
        d = series(E[a], "val/teacher_agreement")
        xs = sorted(d)
        draw(ax1, xs, [d[x] for x in xs], COLOR[a], STYLE[a], LABEL[a], note_below=True)
    ax1.set_xlabel("optimizer step", color=INK2, fontsize=9)
    ax1.set_ylabel("teacher-forced agreement", color=INK2, fontsize=9)
    ax1.set_title("Teacher-forced agreement rises everywhere,\n"
                  "and ranks them differently from held-out decoding",
                  color=INK, fontsize=10, loc="left")

    # Правая панель — валидационный лосс, а НЕ приёмка: приёмка на восьми
    # последовательностях квантована по 1/8 и на графике превращается в шум.
    # Лоссы между вариантами несравнимы (объективы разные), поэтому панель
    # читается по НАКЛОНУ каждой кривой в отдельности: ни одна не вышла на
    # плато, то есть 10k — это равный бюджет, а не сошедшееся состояние.
    for a in runs:
        d = series(E[a], "val/loss")
        xs = sorted(d)
        draw(ax2, xs, [d[x] for x in xs], COLOR[a], STYLE[a])
    ax2.set_xlabel("optimizer step", color=INK2, fontsize=9)
    ax2.set_ylabel("validation loss", color=INK2, fontsize=9)
    ax2.set_title("Nothing has converged at 10k steps:\n"
                  "a matched budget, not a converged state",
                  color=INK, fontsize=10, loc="left")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.text(0.5, 0.0, "Loss curves are not comparable BETWEEN configurations — the objectives "
             "differ. "
             "Read each slope on its own. Held-out acceptance is measured separately.",
             ha="center", va="bottom", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    out = "results/figures/qwen06_training.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
