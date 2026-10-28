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
ARMS = [("qwen06_orthrus", "Orthrus, reproduced (Q,K,V)", SLOT[0], "-"),
        ("qwen06_orthrus_multistep", "masked + multi-step training (Q,K,V,O)", SLOT[1], "-"),
        ("qwen06_flowdraft", "continuous state, no multi-step (Q,K,V,O)", SLOT[2], "-"),
        ("qwen06_flowdraft_multistep", "continuous state + multi-step (Q,K,V,O)", SLOT[3], "-"),
        ("qwen06_flowdraft_multistep_qkv", "continuous state + multi-step (Q,K,V)", SLOT[3], "--")]
PASSES = (1, 2, 3, 4)


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.1), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xticks(PASSES)
        ax.set_xlabel("refinement passes", color=INK2, fontsize=9)

    for key, label, color, style in ARMS:
        if not all(g.get((key, n)) for n in PASSES):
            continue
        acc = [st.mean(x["acceptance"] for x in g[(key, n)].values()) for n in PASSES]
        tpf = [st.mean(x["tpf_steady"] for x in g[(key, n)].values()) for n in PASSES]
        ax1.plot(PASSES, acc, color=color, ls=style, lw=2.2, marker="o", ms=6, label=label)
        ax2.plot(PASSES, tpf, color=color, ls=style, lw=2.2, marker="o", ms=6)
        ax1.annotate(f"{acc[-1]:.2f}", (PASSES[-1], acc[-1]), color=color, fontsize=9,
                     xytext=(6, -3), textcoords="offset points")

    ax1.set_ylabel("accepted tokens per cycle", color=INK2, fontsize=9)
    ax1.set_title("Extra passes keep paying only for the continuous state\ntrained on its own refinement",
                  color=INK, fontsize=10, loc="left")
    ax1.set_xlim(0.8, 4.55)

    ax2.axhline(1.0, color=INK2, lw=1, ls=":")
    ax2.set_ylabel("tokens per forward", color=INK2, fontsize=9)
    ax2.set_title("But none of them pay in throughput: acceptance grows\n"
                  "slower than the forwards it costs",
                  color=INK, fontsize=10, loc="left")

    n_meas = sum(1 for k, *_ in ARMS for n in PASSES if g.get((k, n))) * 6
    n_arms = sum(1 for k, *_ in ARMS if all(g.get((k, n)) for n in PASSES))
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK2, lw=1, ls=":"))
    labels.append("plain autoregressive decoding")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.text(0.5, 0.0, f"Qwen3-0.6B, all {n_arms} configurations at a matched 10k steps. Six benchmarks, "
             f"460 prompts, output identical to greedy AR in all {n_meas} measurements.\n"
             "Four passes use the schedule 0 · 0.34 · 0.67 · 0.9, whose entry 0.34 lies outside the trained range [0.5, 1).",
             ha="center", va="bottom", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    out = "results/figures/qwen06_passes.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
