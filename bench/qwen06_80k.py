"""Qwen3-0.6B при согласованном бюджете 80 000 шагов: три фигуры.

Первая — три панели одной оси «число проходов уточнения»: принятые токены,
токены на проход и ускорение по стенным часам. Третья панель появляется впервые:
прежние замеры шли на маке, где прямой проход упирается в постоянные накладные
расходы и выигрыш в токенах не превращается в секунды. Здесь замер сделан на
T4, и превращается.

Вторая — находка про опору расписания. Точка входа `s = 0.34` штатного
четырёхпроходного расписания лежит ЛЕВЕЕ `selfcorrect_s_min = 0.5`, то есть вне
диапазона, на котором обучалась самокоррекция, и приёмка там рушится. Сдвиг той
же точки внутрь опоры даёт лучший результат кампании. Orthrus служит контролем:
у него нет ни самокоррекции, ни `s_min`, и оба расписания дают у него ОДНО И ТО
ЖЕ число на каждом из 460 промптов.

Третья — разбивка по наборам: несёт ли среднее один датасет. Не несёт.
"""
import json
import math
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
SLOT = ["#8b8a85", "#eb6834", "#eda100", "#2a78d6"]
# Цвет = эксперимент, как во всех фигурах README: серый — воспроизведённый
# Orthrus, синий — непрерывное состояние с многошаговостью. Пунктир у синего
# означает набор проекций Q,K,V без O, а не другой эксперимент.
ARMS = [("orthrus", "Orthrus, reproduced (Q,K,V)", SLOT[0], "-"),
        ("qkv", "continuous state + multi-step (Q,K,V)", SLOT[3], "--")]
FILES = {"orthrus": "kaggle_out/fd80k-eval-orthrus/result/fd80k-eval-orthrus.jsonl",
         "qkv": "kaggle_out/fd80k-eval-qkv/result/fd80k-eval-qkv.jsonl"}
PP = {"orthrus": "kaggle_out/fd80k-eval-orthrus/result/pp-fd80k-eval-orthrus.jsonl",
      "qkv": "kaggle_out/fd80k-eval-qkv/result/pp-fd80k-eval-qkv.jsonl"}
PASSES = (1, 2, 3, 4)
SETS = ["gsm8k", "math500", "humaneval", "mbpp", "aime24", "aime25"]


def sched(r):
    j = r["jumps"]
    if j == 1:
        return "n1"
    return "n4v" if [x[0] for x in j] == [0, 0.5, 0.7, 0.85] else f"n{len(j)}"


def load(path):
    g = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        g[sched(r)].append(r)
    return g


def mean(rows, field):
    return sum(r[field] for r in rows) / len(rows)


def style(ax, xlabel="refinement passes"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def fig_passes():
    g = {k: load(v) for k, v in FILES.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), facecolor=SURFACE)
    for ax in axes:
        style(ax)
        ax.set_xticks(PASSES)
    keys = ["n1", "n2", "n3", "n4v"]
    for arm, label, color, ls in ARMS:
        acc = [mean(g[arm][k], "acceptance") for k in keys]
        tpf = [mean(g[arm][k], "tpf_steady") for k in keys]
        spd = [mean(g[arm][k], "speedup") for k in keys]
        for ax, y in zip(axes, (acc, tpf, spd)):
            ax.plot(PASSES, y, color=color, ls=ls, lw=2.2, marker="o", ms=6,
                    label=label if ax is axes[0] else None)
        axes[0].annotate(f"{acc[-1]:.2f}", (4, acc[-1]), color=color, fontsize=9,
                         xytext=(6, -2), textcoords="offset points")
        # Штатное расписание с точкой входа вне обученной опоры — полая метка:
        # это та же модель и то же число проходов, отличается только расписание.
        broken = mean(g[arm]["n4"], "acceptance")
        axes[0].plot([4], [broken], marker="o", ms=7, mfc=SURFACE, mec=color, mew=2, ls="none")
    axes[0].annotate("schedule leaving the trained range",
                     (4, mean(g["qkv"]["n4"], "acceptance")), color=INK2, fontsize=8,
                     xytext=(-8, -16), textcoords="offset points", ha="right")
    axes[0].set_ylabel("accepted tokens per cycle", color=INK2, fontsize=9)
    axes[0].set_title("the flow map's acceptance grows with passes",
                      color=INK, fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    for ax, name, title in ((axes[1], "tokens per forward",
                             "extra passes still cost more than they return"),
                            (axes[2], "wall-clock speedup over plain decoding",
                             "and on a real GPU the gain reaches seconds")):
        ax.axhline(1.0, color=INK2, lw=1, ls=":")
        ax.set_ylabel(name, color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[1].annotate("autoregressive decoding", (3.62, 1.0), color=INK2, fontsize=8,
                     ha="right", va="bottom", xytext=(0, 3), textcoords="offset points")
    fig.text(0.5, -0.10,
             "Qwen3-0.6B, both drafters at a matched 80,000 optimizer steps · 460 prompts from six benchmarks · "
             "every measurement bitwise-identical to greedy AR\n"
             "hollow marker: the four-pass schedule whose entry s = 0.34 falls outside the trained range [0.5, 1)",
             ha="center", color=INK2, fontsize=8.5)
    out = "results/figures/qwen06_80k_passes.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("написано", out)


def fig_schedule():
    g = {k: load(v) for k, v in FILES.items()}
    fig, ax = plt.subplots(figsize=(8.2, 4.3), facecolor=SURFACE)
    style(ax, xlabel=None)
    keys = [("n3", "three passes\n0 · 0.5 · 0.75"),
            ("n4", "four passes\n0 · 0.34 · 0.67 · 0.9"),
            ("n4v", "four passes\n0 · 0.5 · 0.70 · 0.85")]
    x = range(len(keys))
    w = 0.36
    for i, (arm, label, color, _) in enumerate(ARMS):
        vals = [mean(g[arm][k], "acceptance") for k, _ in keys]
        pos = [p + (i - 0.5) * w for p in x]
        ax.bar(pos, vals, width=w, color=color, label=label, zorder=3)
        for p, v in zip(pos, vals):
            ax.annotate(f"{v:.2f}", (p, v), color=INK2, fontsize=9, ha="center",
                        xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, 5.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in keys], color=INK2, fontsize=9)
    ax.set_ylabel("accepted tokens per cycle", color=INK2, fontsize=9)
    ax.set_title("one entry point outside the trained range costs more than a whole pass buys",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.text(0.5, -0.14,
             "Training places the two refinement entries in [0.5, 0.75) and [0.75, 1); nothing left of 0.5 is ever seen.\n"
             "Orthrus is the control: it has no self-correction and no s_min, and its two four-pass bars are\n"
             "identical on every one of the 460 prompts.",
             ha="center", color=INK2, fontsize=8.5)
    out = "results/figures/qwen06_80k_schedule.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("написано", out)


def fig_benchmarks():
    """Уровни ОБЕИХ веток по наборам, а не только их разность.

    Первая версия этой фигуры рисовала лишь контраст, и по ней нельзя было
    увидеть, откуда он берётся: столбец «+1.29» не говорит, 2.2 против 3.5 это
    или 1.0 против 2.3. Здесь стоят оба уровня, как в фигуре того же названия
    для раздела на 10 000 шагов, а разность подписана над парой.
    """
    vals = defaultdict(lambda: defaultdict(list))
    for arm, path in PP.items():
        for line in open(path):
            r = json.loads(line)
            if sched(r) != "n3":
                continue
            vals[arm][r["dataset"]].append(r["acceptance"])
    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor=SURFACE)
    style(ax, xlabel=None)
    x = list(range(len(SETS)))
    w = 0.36
    for i, (arm, label, color, _) in enumerate(ARMS):
        ys = [sum(vals[arm][d]) / len(vals[arm][d]) for d in SETS]
        es = [1.96 * st.stdev(vals[arm][d]) / math.sqrt(len(vals[arm][d])) for d in SETS]
        pos = [p + (i - 0.5) * w for p in x]
        ax.bar(pos, ys, width=w, color=color, label=label, zorder=3,
               yerr=es, capsize=2.5, error_kw={"ecolor": INK2, "lw": 1.1})
    top = 0
    for j, d in enumerate(SETS):
        a = sum(vals["orthrus"][d]) / len(vals["orthrus"][d])
        b = sum(vals["qkv"][d]) / len(vals["qkv"][d])
        e = 1.96 * st.stdev(vals["qkv"][d]) / math.sqrt(len(vals["qkv"][d]))
        top = max(top, b + e)
        # Подпись ставится над усом, а не над столбцом: на малых наборах ус
        # длиннее разницы между ними, и подпись на него наезжала.
        ax.annotate(f"+{b - a:.2f}", (j, b + e), color=INK, fontsize=9.5, ha="center",
                    xytext=(0, 8), textcoords="offset points", fontweight="bold")
    ax.set_ylim(0, top * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={len(vals['qkv'][d])})" for d in SETS],
                       color=INK2, fontsize=9)
    ax.set_ylabel("accepted tokens per cycle", color=INK2, fontsize=9)
    ax.set_title("three refinement passes: the margin is on every benchmark, not one",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.text(0.5, -0.12,
             "Both drafters at a matched 80,000 steps. Bold numbers are the paired margin, computed\n"
             "prompt by prompt. Whiskers are 95% intervals over prompts within a benchmark, so they\n"
             "show measurement spread, not the between-seed spread the 135M figures carry.",
             ha="center", color=INK2, fontsize=8.5)
    out = "results/figures/qwen06_80k_per_benchmark.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("написано", out)


if __name__ == "__main__":
    fig_passes()
    fig_schedule()
    fig_benchmarks()
