"""Acceptance during training, Qwen3-0.6B — the companion to the SmolLM figure.

Two panels, because the in-training number and the measured one disagree and
the disagreement is the point.

Left: acceptance as the training loop sees it — the sum of the per-position
prefix conjunctions logged at each validation. It reads EIGHT held-out
sequences, so it is quantised to 1/8 and noisy; it is an indicator of the
trajectory, not a measurement.

Right: acceptance as decoding actually delivers it, measured after training on
460 prompts across six benchmarks. Same quantity, three orders of magnitude
more evidence, and a different ordering.

There is no seed band: Qwen has one training seed, unlike the 135M bench.
The Q,K,V run has a single in-training point because its earlier validations
were lost with the output directory it shared between sessions.
"""
import glob
import json
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
SLOT = ["#8b8a85", "#eb6834", "#eda100", "#2a78d6"]
RUNS = [("qwen06_orthrus", "Orthrus (Q,K,V)", SLOT[0], "-"),
        ("qwen06_orthrus_multistep", "Orthrus multistep (Q,K,V,O)", SLOT[1], "-"),
        ("qwen06_flowdraft", "FlowDraft (Q,K,V,O)", SLOT[2], "-"),
        ("qwen06_flowdraft_multistep", "FlowDraft multistep (Q,K,V,O)", SLOT[3], "-"),
        ("qwen06_flowdraft_multistep_qkv", "FlowDraft multistep (Q,K,V)", SLOT[3], "--")]
# The five runs form two matched sets, and a panel compares runs of one set only:
# three on Q,K,V,O with the same position weights, which differ in nothing but
# the state and the multistep term, and the Q,K,V pair of Orthrus against
# FlowDraft multistep, the projections the Orthrus paper describes.
SETS = [("Q,K,V,O, position weights in all three",
         ["qwen06_orthrus_multistep", "qwen06_flowdraft", "qwen06_flowdraft_multistep"]),
        ("Q,K,V, as in the Orthrus paper",
         ["qwen06_orthrus", "qwen06_flowdraft_multistep_qkv"])]
BLOCK = 31


def in_training():
    S = defaultdict(dict)
    for ev in glob.glob("kaggle_out/*/qwen06_*/lightning_logs/version_*/events*"):
        if "r1_timeout" in ev:
            continue
        exp = ev.split("/")[2]
        ea = EventAccumulator(ev, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        for j in range(1, 32):
            t = f"val/acceptance_pos_{j:02d}"
            if t in tags:
                for s in ea.Scalars(t):
                    S[exp].setdefault(s.step, 0.0)
                    S[exp][s.step] += s.value
    return S


def measured():
    """Среднее по СРЕДНИМ ШЕСТИ НАБОРОВ, а не по промптам.

    Наборы разного размера — в aime по 30 задач против 100 в остальных, — и
    усреднение по промптам молча взвесило бы их по объёму. Таблицы README
    считают равновесно по наборам, и панель обязана считать так же, иначе
    подпись и картинка разойдутся.
    """
    per = defaultdict(lambda: defaultdict(dict))
    for line in open("results/qwen06-10k.jsonl"):
        r = json.loads(line)
        j = r.get("jumps")
        n = 1 if isinstance(j, (int, float)) else len(j)
        per[r["checkpoint"].split("/")[-2]][n][r["dataset"]] = r["acceptance"]
    return per


def draw(ax, xs, ys, color, style, label=None):
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
                    color=color, fontsize=7.5, ha="right", va="center",
                    xytext=(-14, 0), textcoords="offset points")
    else:
        ax.plot(xs, ys, color=color, ls=style, lw=2, marker="o", ms=5, label=label)


def main():
    S, M = in_training(), measured()
    by_key = {k: (label, color, style) for k, label, color, style in RUNS}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    passes = (1, 2, 3, 4)
    for row, (set_name, keys) in enumerate(SETS):
        ax1, ax2 = axes[row]
        for key in keys:
            label, color, style = by_key[key]
            d = S.get(key, {})
            if d:
                xs = sorted(d)
                draw(ax1, xs, [d[x] for x in xs], color, style, label)
            v = M.get(key)
            if v and all(v.get(n) for n in passes):
                ax2.plot(passes, [st.mean(v[n].values()) for n in passes], color=color, ls=style,
                         lw=2.2, marker="o", ms=6, label=label)
        ax1.set_xlabel("training step", color=INK2, fontsize=9)
        ax1.set_ylabel(f"accepted tokens out of {BLOCK}", color=INK2, fontsize=9)
        ax1.set_title(f"({'ac'[row]}) {set_name}: as the training loop sees it\n"
                      "eight held-out sequences, quantised to 1/8", color=INK, fontsize=10, loc="left")
        ax2.set_xticks(passes)
        ax2.set_xlabel("refinement passes at decode", color=INK2, fontsize=9)
        ax2.set_ylabel(f"accepted tokens out of {BLOCK}", color=INK2, fontsize=9)
        ax2.set_title(f"({'bd'[row]}) {set_name}: as decoding delivers it\n"
                      "460 prompts, six benchmarks", color=INK, fontsize=10, loc="left")
        ax2.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.text(0.5, 0.0, "Qwen3-0.6B at a matched 10k steps, one training seed — no seed band, "
             "unlike the 135M bench. The left panels are an indicator; the right ones are the measurement.",
             ha="center", va="bottom", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = "results/figures/qwen06_acceptance.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
