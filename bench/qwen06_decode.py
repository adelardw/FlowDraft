"""Second scale point, Qwen3-0.6B: decode measurements at a matched 10k steps.

Two panels, two different questions, and the conclusion needs both.

Left: accepted tokens against the number of refinement passes. This is where
the method's signature lives — extra passes keep paying only for the drafter
trained on its own refinement; the other configurations are flat.

Right: the same axis in throughput, `(accepted + 1) / (passes + 1)`. It FALLS
everywhere: acceptance grows slower than the forwards it costs. The unit
line is plain autoregressive decoding; below it a multi-pass schedule is worse
than not drafting at all. Showing the left panel without the right one would
misrepresent the result.
"""
import json
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
SLOT = ["#8b8a85", "#eb6834", "#eda100", "#2a78d6"]
# Порядок и цвета — общие со всеми фигурами README: один цвет = один
# эксперимент. Маскирующее + многошаг появляется, только когда оно замерено:
# эта арматура доучивалась последней.
# Пятая линия — та же конфигурация без o_proj, а не отдельный эксперимент,
# поэтому цвет тот же, а отличается стиль. Это согласуется с остальными
# фигурами README, где цвет = эксперимент.
ARMS = [("qwen06_orthrus", "Orthrus (Q,K,V)", SLOT[0], "-"),
        ("qwen06_orthrus_multistep", "Orthrus multistep (Q,K,V,O)", SLOT[1], "-"),
        ("qwen06_flowdraft", "FlowDraft (Q,K,V,O)", SLOT[2], "-"),
        ("qwen06_flowdraft_multistep", "FlowDraft multistep (Q,K,V,O)", SLOT[3], "-"),
        ("qwen06_flowdraft_multistep_qkv", "FlowDraft multistep (Q,K,V)", SLOT[3], "--")]
PASSES = (1, 2, 3, 4)
# The five runs form two matched sets, and a panel compares runs of one set only:
# three on Q,K,V,O with the same position weights, which differ in nothing but
# the state and the multistep term, and the Q,K,V pair of Orthrus against
# FlowDraft multistep, the projections the Orthrus paper describes.
SETS = [("Q,K,V,O, position weights in all three",
         ["qwen06_orthrus_multistep", "qwen06_flowdraft", "qwen06_flowdraft_multistep"]),
        ("Q,K,V, as in the Orthrus paper",
         ["qwen06_orthrus", "qwen06_flowdraft_multistep_qkv"])]


def load(path="results/qwen06-10k.jsonl"):
    g = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        j = r.get("jumps")
        n = 1 if isinstance(j, (int, float)) else len(j)
        g[(r["checkpoint"].split("/")[-2], n)][r["dataset"]] = r
    return g


def main():
    g = load()
    by_key = {k: (label, color, style) for k, label, color, style in ARMS}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xticks(PASSES)
        ax.set_xlabel("refinement passes", color=INK2, fontsize=9)
    n_meas = 0
    for row, (set_name, keys) in enumerate(SETS):
        ax1, ax2 = axes[row]
        for key in keys:
            label, color, style = by_key[key]
            if not all(g.get((key, n)) for n in PASSES):
                continue
            n_meas += 4 * 6
            acc = [st.mean(x["acceptance"] for x in g[(key, n)].values()) for n in PASSES]
            tpf = [st.mean(x["tpf_steady"] for x in g[(key, n)].values()) for n in PASSES]
            ax1.plot(PASSES, acc, color=color, ls=style, lw=2.2, marker="o", ms=6, label=label)
            ax2.plot(PASSES, tpf, color=color, ls=style, lw=2.2, marker="o", ms=6, label=label)
            ax1.annotate(f"{acc[-1]:.2f}", (PASSES[-1], acc[-1]), color=color, fontsize=9,
                         xytext=(6, -3), textcoords="offset points")
        ax1.set_xlim(0.8, 4.55)
        ax1.set_ylabel("accepted tokens per cycle", color=INK2, fontsize=9)
        ax1.set_title(f"({'ac'[row]}) {set_name}: acceptance", color=INK, fontsize=10, loc="left")
        ax2.axhline(1.0, color=INK2, lw=1, ls=":")
        ax2.set_ylabel("tokens per forward", color=INK2, fontsize=9)
        ax2.set_title(f"({'bd'[row]}) {set_name}: tokens per forward", color=INK, fontsize=10, loc="left")
        ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.text(0.5, 0.0, f"Qwen3-0.6B, five runs at a matched 10k steps in two matched sets; one training seed. "
             f"Six benchmarks, 460 prompts, output identical to greedy AR in all {n_meas} measurements.\n"
             "Extra passes pay in acceptance only for FlowDraft multistep and pay in throughput for none; "
             "the dotted line is plain autoregressive decoding.\n"
             "Four passes use the schedule 0 · 0.34 · 0.67 · 0.9, whose entry 0.34 lies outside the trained range [0.5, 1).",
             ha="center", va="bottom", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = "results/figures/qwen06_passes.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
