"""Per-benchmark breakdown for the Qwen3-0.6B scale point.

The companion to the SmolLM figure of the same name: does the effect hold on
every benchmark, or is the average carried by one of them? Error bars are 95%
intervals over prompts within each benchmark, so they show measurement spread,
not between-seed spread — Qwen has one training seed.

Four refinement passes, which is where the multi-step contrast is largest.
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
RUNS = [("qwen06_orthrus", "Orthrus, reproduced (Q,K,V)", SLOT[0], None),
        ("qwen06_orthrus_multistep", "masked + multi-step training (Q,K,V,O)", SLOT[1], None),
        ("qwen06_flowdraft", "continuous state, no multi-step (Q,K,V,O)", SLOT[2], None),
        ("qwen06_flowdraft_multistep", "continuous + multi-step (Q,K,V,O)", SLOT[3], None),
        ("qwen06_flowdraft_multistep_qkv", "continuous + multi-step (Q,K,V)", SLOT[3], "//")]
PASSES = 4
BLOCK = 31
ORDER = ["gsm8k", "math500", "humaneval", "mbpp", "aime24", "aime25"]


def main():
    per = defaultdict(list)
    for line in open("results/pp-qwen06-10k.jsonl"):
        r = json.loads(line)
        j = r.get("jumps")
        n = 1 if isinstance(j, (int, float)) else len(j)
        if n != PASSES:
            continue
        per[(r["checkpoint"].split("/")[-2], r["dataset"])].append(r["acceptance"])

    benches = [b for b in ORDER if any(d == b for _, d in per)]
    present = [x for x in RUNS if any((x[0], b) in per for b in benches)]
    width = 0.8 / len(present)

    fig, ax = plt.subplots(figsize=(12.5, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)

    for i, (key, label, color, hatch) in enumerate(present):
        xs, ys, es = [], [], []
        for j, b in enumerate(benches):
            v = per.get((key, b), [])
            if not v:
                continue
            xs.append(j - 0.4 + width * (i + 0.5))
            ys.append(st.mean(v))
            es.append(1.96 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.bar(xs, ys, width * 0.9, yerr=es, capsize=2.5, label=label,
               color=color, hatch=hatch, edgecolor=SURFACE, linewidth=0.6,
               error_kw={"ecolor": INK2, "elinewidth": 1})

    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches, color=INK2, fontsize=9.5)
    ax.set_ylabel(f"accepted tokens out of {BLOCK}", color=INK2, fontsize=9)
    ax.set_title(f"The effect holds on every benchmark, {PASSES} refinement passes (entry s = 0.34, outside the trained range)",
                 color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, 5.0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper left")
    fig.text(0.5, -0.02, "Qwen3-0.6B at a matched 10k steps. Bars are means over prompts, "
             "whiskers are 95% intervals over prompts within the benchmark — "
             "measurement spread, not between-seed spread.",
             ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout()
    out = "results/figures/qwen06_per_benchmark.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
