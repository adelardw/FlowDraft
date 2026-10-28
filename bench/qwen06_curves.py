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
    ("qwen06_orthrus", "Orthrus (Q,K,V)", SLOT[0], "-"),
    ("qwen06_orthrus_multistep", "Orthrus multistep (Q,K,V,O)", SLOT[1], "-"),
    ("qwen06_flowdraft", "FlowDraft (Q,K,V,O)", SLOT[2], "-"),
    ("qwen06_flowdraft_multistep", "FlowDraft multistep (Q,K,V,O)", SLOT[3], "-"),
    ("qwen06_flowdraft_multistep_qkv", "FlowDraft multistep (Q,K,V)", SLOT[3], "--"),
]
ARMS = [(k, l) for k, l, _, _ in RUNS]
COLOR = {k: c for k, _, c, _ in RUNS}
STYLE = {k: st for k, _, _, st in RUNS}
LABEL = dict(ARMS)
# The five runs form two matched sets, and a panel compares runs of one set only:
# three on Q,K,V,O with the same position weights, which differ in nothing but
# the state and the multistep term, and the Q,K,V pair of Orthrus against
# FlowDraft multistep, the projections the Orthrus paper describes.
SETS = [("Q,K,V,O, position weights in all three",
         ["qwen06_orthrus_multistep", "qwen06_flowdraft", "qwen06_flowdraft_multistep"]),
        ("Q,K,V, as in the Orthrus paper",
         ["qwen06_orthrus", "qwen06_flowdraft_multistep_qkv"])]
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
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xlabel("optimizer step", color=INK2, fontsize=9)
    for row, (set_name, keys) in enumerate(SETS):
        ax1, ax2 = axes[row]
        for a in [k for k in keys if k in E]:
            d = series(E[a], "val/teacher_agreement")
            xs = sorted(d)
            draw(ax1, xs, [d[x] for x in xs], COLOR[a], STYLE[a], LABEL[a], note_below=False)
            # Validation loss, not acceptance: acceptance on eight sequences is
            # quantised to 1/8. The objectives differ, so read each slope alone.
            d = series(E[a], "val/loss")
            xs = sorted(d)
            draw(ax2, xs, [d[x] for x in xs], COLOR[a], STYLE[a])
        ax1.set_ylabel("teacher-forced agreement", color=INK2, fontsize=9)
        ax1.set_title(f"({'ac'[row]}) {set_name}: teacher-forced agreement", color=INK, fontsize=10, loc="left")
        ax2.set_ylabel("validation loss", color=INK2, fontsize=9)
        ax2.set_title(f"({'bd'[row]}) {set_name}: validation loss", color=INK, fontsize=10, loc="left")
        ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right" if row == 0 else "upper left")
    fig.text(0.5, 0.0, "Qwen3-0.6B at a matched 10k steps. Loss curves are not comparable BETWEEN configurations — "
             "the objectives differ; read each slope on its own. Nothing has converged at 10k steps.",
             ha="center", va="bottom", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = "results/figures/qwen06_training.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
