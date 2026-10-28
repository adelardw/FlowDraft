"""The four SmolLM2-135M figures of the README, rebuilt from the per-prompt data.

    .venv/bin/python bench/smollm_figures.py

Writes results/figures/{multistep,contrasts,per_benchmark,horizon}.png from
data/measurements/2026-08-09-per-prompt.jsonl.gz (seed 42, every snapshot) and
data/measurements/2026-08-10-seeds-per-prompt.jsonl.gz (seeds 43 and 44, 20k).
The 20k numbers average the three seeds — per-seed means pooled over the 460
prompts — and the intervals use the seed as the unit (Student t, df = 2).
Tokens per forward is the end-to-end rate, prefill included, as in the README's
135M section. The horizon figure is seed 42 alone: it is the only seed
measured at 10k and 16k.
"""
import gzip
import json
import math
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
SLOT = ["#8b8a85", "#eb6834", "#eda100", "#2a78d6"]
NEG = SLOT[1]
ALL = [("smollm_orthrus", "Orthrus (Q,K,V)"),
       ("smollm_orthrus_multistep", "Orthrus multistep (Q,K,V,O)"),
       ("smollm_flowdraft", "FlowDraft (Q,K,V,O)"),
       ("smollm_flowdraft_multistep", "FlowDraft multistep (Q,K,V,O)")]
COLOR = {c: SLOT[i] for i, (c, _) in enumerate(ALL)}
# The figures compare only runs that are matched: the three that train Q,K,V,O
# with the same position weights and differ in nothing but the state and the
# multistep term. Orthrus trains Q,K,V without the weights, so it is matched
# to none of them and is left out (its numbers are in the README text).
CONFIGS = ALL[1:]
SCHED = {"1": 1, "[[0, 1], [0.5, 1], [0.75, 1]]": 3, "[[0, 1], [0.34, 1], [0.67, 1], [0.9, 1]]": 4}
PASSES = [1, 3, 4]
SEEDS = [42, 43, 44]
DATASETS = ["gsm8k", "math500", "humaneval", "mbpp", "aime24", "aime25"]
T2 = stats.t.ppf(0.975, 2)

# (config, seed, step, passes, dataset) -> {prompt index: row}
ROWS = defaultdict(dict)


def load(path, seed=None):
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        n = SCHED.get(json.dumps(r["jumps"]))
        if n is None:
            continue
        key = (r["checkpoint"].split("/")[-2], seed or r["training_seed"], r["checkpoint_step"], n, r["dataset"])
        ROWS[key][r["prompt_index"]] = r


load("data/measurements/2026-08-09-per-prompt.jsonl.gz", seed=42)
load("data/measurements/2026-08-10-seeds-per-prompt.jsonl.gz")


def values(cfg, seed, n, metric, step=20000, datasets=DATASETS):
    return [r[metric] for ds in datasets for r in ROWS[(cfg, seed, step, n, ds)].values()]


def seed_means(cfg, n, metric, datasets=DATASETS):
    return [st.mean(values(cfg, s, n, metric, datasets=datasets)) for s in SEEDS]


def seed_ci(xs):
    return st.mean(xs), T2 * st.stdev(xs) / math.sqrt(len(xs))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=INK2, labelsize=9)


def multistep():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURFACE)
    for ax, metric, ylabel, title in (
        (axes[0], "acceptance", "accepted tokens out of 31",
         "Acceptance grows with refinement passes only for\nFlowDraft multistep"),
        (axes[1], "tpf", "tokens per forward, end to end", "Every extra pass costs throughput"),
    ):
        style(ax)
        ax.axvspan(3.6, 4.4, color="#f0efeb", zorder=0, lw=0)
        for cfg, label in CONFIGS:
            ms, hs = zip(*(seed_ci(seed_means(cfg, n, metric)) for n in PASSES))
            ax.errorbar(PASSES, ms, yerr=hs, color=COLOR[cfg], marker="o", ms=6, lw=2.2, capsize=4,
                        markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
        ax.set_xticks(PASSES)
        ax.set_xticklabels(["1", "3", "4\nentry 0.34,\noutside training"])
        ax.set_xlim(0.7, 4.4)
        ax.set_xlabel("refinement passes per cycle", color=INK2, fontsize=10)
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[1].axhline(1.0, color=INK2, lw=1.0, zorder=1)
    axes[1].text(3.5, 1.02, "plain autoregressive decoding", ha="right", va="bottom", fontsize=8.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("SmolLM2-135M · the three configurations on Q,K,V,O with position weights · 460 tasks · "
                 "three training seeds · error bars are seed-to-seed 95% CI",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig("results/figures/multistep.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("-> results/figures/multistep.png")


def contrasts():
    rows = [("smollm_flowdraft_multistep", "smollm_flowdraft", "multistep term\nFlowDraft multistep − FlowDraft"),
            ("smollm_flowdraft_multistep", "smollm_orthrus_multistep", "continuous state\nFlowDraft multistep − Orthrus multistep")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 2.9), facecolor=SURFACE, sharey=True)
    for ax, n in zip(axes, PASSES):
        style(ax)
        ax.grid(axis="y", visible=False)
        for y, (a, b, _) in enumerate(rows):
            d = [x - z for x, z in zip(seed_means(a, n, "acceptance"), seed_means(b, n, "acceptance"))]
            m, h = seed_ci(d)
            color = SLOT[3] if m > 0 else NEG
            ax.errorbar([m], [y], xerr=[h], color=color, marker="o", ms=6, lw=2.4, capsize=4,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ax.text(m + h + 0.02 * (1 if n == 1 else 3), y, f"{m:+.2f}", va="center", fontsize=9, color=INK2)
        ax.axvline(0, color=INK2, lw=1.0, zorder=1)
        ax.set_xlabel("Δ accepted tokens", color=INK2, fontsize=10)
        title = f"{n} refinement pass" + ("es" if n > 1 else "") + (" (entry 0.34)" if n == 4 else "")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[2] for r in rows], color=INK2, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Paired contrasts with the SEED as the unit of observation (df = 2) · 20k steps · each changes one "
                 "factor: both runs train Q,K,V,O with the same position weights",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("results/figures/contrasts.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("-> results/figures/contrasts.png")


def per_benchmark():
    fig, ax = plt.subplots(figsize=(12.5, 5.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", visible=False)
    w = 0.26
    for i, (cfg, label) in enumerate(CONFIGS):
        for j, ds in enumerate(DATASETS):
            m, h = seed_ci(seed_means(cfg, 3, "acceptance", datasets=[ds]))
            ax.bar(j + (i - 1) * w, m, w * 0.92, color=COLOR[cfg], label=label if j == 0 else None, zorder=3)
            ax.errorbar(j + (i - 1) * w, m, yerr=h, color=INK2, lw=1.2, capsize=3, zorder=4)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, color=INK2, fontsize=10)
    ax.set_ylabel("accepted tokens out of 31", color=INK2, fontsize=10)
    ax.set_title("The effect holds on every benchmark, three refinement passes", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, ncol=3, loc="upper left")
    ax.set_ylim(0, 3.3)
    fig.text(0.5, -0.02, "SmolLM2-135M at 20k steps, three training seeds, the three configurations on Q,K,V,O. "
             "Bars are seed means; whiskers are 95% intervals with the seed as the unit.", ha="center", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig("results/figures/per_benchmark.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("-> results/figures/per_benchmark.png")


def horizon():
    steps = [10000, 16000, 20000]
    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor=SURFACE)
    style(ax)
    for cfg, label in CONFIGS:
        ms, hs = [], []
        for step in steps:
            v = values(cfg, 42, 3, "acceptance", step=step)
            ms.append(st.mean(v))
            hs.append(1.96 * st.stdev(v) / math.sqrt(len(v)))
        ax.errorbar([s / 1000 for s in steps], ms, yerr=hs, color=COLOR[cfg], marker="o", ms=6, lw=2.4,
                    capsize=4, markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
    ax.set_xticks([s / 1000 for s in steps])
    ax.set_xlabel("training steps, thousands", color=INK2, fontsize=10)
    ax.set_ylabel("accepted tokens out of 31, three passes", color=INK2, fontsize=10)
    ax.set_title("Measured horizon: eight of ten runs peak at the budget limit", color=INK, fontsize=12, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK2)
    fig.text(0.5, 0.0, "Seed 42, the only seed measured at 10k and 16k; the three configurations on Q,K,V,O of the ten. "
             "Whiskers are 95% intervals over the 460 prompts.", ha="center", va="bottom", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig("results/figures/horizon.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("-> results/figures/horizon.png")


if __name__ == "__main__":
    multistep()
    contrasts()
    per_benchmark()
    horizon()
