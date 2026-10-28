"""Per-benchmark breakdown for the Qwen3-0.6B scale point.

The companion to the SmolLM figure of the same name: does the effect hold on
every benchmark, or is the average carried by one of them? Error bars are 95%
intervals over prompts within each benchmark, so they show measurement spread,
not between-seed spread — Qwen has one training seed.

Three refinement passes, the schedule 0 · 0.5 · 0.75 that stays inside the trained range.
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
RUNS = [("qwen06_orthrus", "Orthrus (Q,K,V)", SLOT[0], None),
        ("qwen06_orthrus_multistep", "Orthrus multistep (Q,K,V,O)", SLOT[1], None),
        ("qwen06_flowdraft", "FlowDraft (Q,K,V,O)", SLOT[2], None),
        ("qwen06_flowdraft_multistep", "FlowDraft multistep (Q,K,V,O)", SLOT[3], None),
        ("qwen06_flowdraft_multistep_qkv", "FlowDraft multistep (Q,K,V)", SLOT[3], "//")]
PASSES = 3
# The five runs form two matched sets, and a panel compares runs of one set only:
# three on Q,K,V,O with the same position weights, which differ in nothing but
# the state and the multistep term, and the Q,K,V pair of Orthrus against
# FlowDraft multistep, the projections the Orthrus paper describes.
SETS = [("Q,K,V,O, position weights in all three",
         ["qwen06_orthrus_multistep", "qwen06_flowdraft", "qwen06_flowdraft_multistep"]),
        ("Q,K,V, as in the Orthrus paper",
         ["qwen06_orthrus", "qwen06_flowdraft_multistep_qkv"])]
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
    by_key = {k: (label, color, hatch) for k, label, color, hatch in RUNS}
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.0), facecolor=SURFACE)
    for row, (ax, (set_name, keys)) in enumerate(zip(axes, SETS)):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        width = 0.8 / len(keys)
        for i, key in enumerate(keys):
            label, color, hatch = by_key[key]
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
        ax.set_title(f"({'ab'[row]}) {set_name}: {PASSES} refinement passes by benchmark",
                     color=INK, fontsize=11, loc="left")
        ax.set_ylim(0, 4.6)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper left")
    fig.text(0.5, -0.01, "Qwen3-0.6B at a matched 10k steps, two matched sets. Bars are means over prompts, "
             "whiskers are 95% intervals over prompts within the benchmark — "
             "measurement spread, not between-seed spread.",
             ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout()
    out = "results/figures/qwen06_per_benchmark.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
