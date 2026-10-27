"""Training curves per metric, per loss term, per seed, per experiment.

Reads the TensorBoard scalars Lightning writes under
``checkpoints/[s43|s44/]<configuration>/lightning_logs/version_0`` and produces

  results/figures/curves_loss.png     — every loss term that was logged
  results/figures/curves_metrics.png  — every validation metric that was logged
  results/figures/curves_seeds.png    — acceptance per experiment, one panel each,
                                        with the seed-to-seed range shaded

Encoding: colour is the EXPERIMENT, line style is the SEED (solid 42, dashed 43,
dotted 44). Two channels rather than one, so the two questions the figures exist
to answer stay separable — "do the experiments differ" reads off colour, "do the
seeds agree" reads off how tightly the three styles of one colour overlap.

Panels are dropped, never faked: a term logged by only the flow configurations shows only
those configurations, and the panel title says how many runs it covers. A seed that never
ran leaves no line.

    uv run python bench/curves.py
"""
import glob
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e7e3"
# Палитра ОБЩАЯ с multistep/per_benchmark/horizon: один цвет = один
# эксперимент во всех фигурах README. Раньше здесь стоял свой набор, и синий
# означал Orthrus тут и «непрерывное + многошаг» там — прямое противоречие на
# одной странице. Серый у бейзлайна намеренно: это опорная линия, не равный ряд.
SLOT = ["#8b8a85", "#eb6834", "#eda100", "#2a78d6"]

# Fixed order — a run that is absent must not repaint the runs that remain.
ARMS = [
    ("smollm_orthrus", "Orthrus, reproduced (Q,K,V)"),
    ("smollm_orthrus_multistep", "masked + multi-step training (Q,K,V,O)"),
    ("smollm_flowdraft", "continuous state, no multi-step training (Q,K,V,O)"),
    ("smollm_flowdraft_multistep", "continuous state + multi-step training (Q,K,V,O)"),
]
COLOR = {a: SLOT[i] for i, (a, _) in enumerate(ARMS)}
LABEL = dict(ARMS)
# Короткие имена для заголовка панели. Метрики логируются НЕСИММЕТРИЧНО:
# `val/accepted_tokens_per_block` пишет только маскирующая ветвь, а `val/loss/*`
# — только симплексная. Панель, где половина экспериментов молча отсутствует,
# читается как «эти линии совпали», поэтому неполное покрытие называется прямо.
SHORT = {
    "smollm_orthrus": "Orthrus",
    "smollm_orthrus_multistep": "masked+MS",
    "smollm_flowdraft": "continuous",
    "smollm_flowdraft_multistep": "continuous+MS",
}

SEEDS = [(42, "-", ""), (43, "--", "s43"), (44, ":", "s44")]
STYLE = {sd: st for sd, st, _ in SEEDS}

LOSS_PANELS = [
    ("train/loss", "Training loss (rolling median)"),
    ("val/loss", "Validation loss"),
    ("val/loss/verify_kl", "verify KL — first decode refinement pass"),
    ("val/loss/selfcorrect_kl", "multi-step KL — later refinement passes"),
    ("val/loss/endpoint", "endpoint"),
    ("val/loss/ec", "endpoint consistency"),
    ("val/loss/td", "temporal difference"),
    ("val/loss/accepted", "accepted-length surrogate"),
]

METRIC_PANELS = [
    ("val/acceptance_decode", "Accepted tokens per cycle"),
    ("val/tpf", "Tokens per forward"),
    ("val/teacher_agreement", "Agreement with the frozen verifier"),
    ("val/decode/accepted_mean", "Accepted, decode mean"),
    ("val/accepted_tokens_per_block", "Accepted tokens per block"),
    ("lr-AdamW", "Learning rate"),
]


def run_dir(configuration, seed):
    stem = "checkpoints" if seed == 42 else f"checkpoints/s{seed}"
    hits = glob.glob(str(ROOT / stem / configuration / "lightning_logs" / "version_*"))
    return sorted(hits)[-1] if hits else None


def load_run(path):
    """All non-per-position scalars of one run as ``tag -> (steps, values)``."""
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        if re.search(r"_(cycle|pos)_\d+$", tag):
            continue
        pts = ea.Scalars(tag)
        out[tag] = ([p.step for p in pts], [p.value for p in pts])
    return out


def smooth(xs, ys, frac=0.02):
    """Rolling mean over ``frac`` of the series — `train/loss` is logged every
    step, so 20k points of per-batch noise are pure hash at figure size. The
    window scales with length instead of being fixed, otherwise a short run and
    a long one get visually different amounts of smoothing and look like they
    differ in variance. Validation series are short and pass through untouched."""
    window = max(51, int(len(ys) * frac) | 1)
    if len(ys) < window * 2:
        return xs, ys
    half = window // 2
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        # Median, not mean: per-batch loss carries deep downward excursions on
        # short packed sequences, where few positions survive padding. A mean
        # is dragged by them and reports a trend the bulk of the batches never
        # had; the median reports where the batches actually sit.
        out.append(statistics.median(ys[lo:hi]))
    return xs, out


def draw(panels, runs, path, title, subtitle):
    live, flat = [], []
    for tag, name in panels:
        series = [(a, s, runs[(a, s)][tag]) for (a, s) in runs if tag in runs[(a, s)]]
        if not series:
            continue
        # A term whose weight is zero logs an exact zero on every step in every
        # run. Its panel is a flat line on an invented +-0.04 axis, which reads
        # like data and is not. Name it in the caption instead of drawing it.
        vals = [v for _, _, (_, ys) in series for v in ys]
        if max(vals) == min(vals) == 0.0:
            flat.append(name)
        else:
            live.append((tag, name, series))
    if not live:
        print(f"нет данных ни по одной панели -> {path} не построен")
        return

    ncol = 2 if len(live) <= 4 else 3
    nrow = math.ceil(len(live) / ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow), facecolor=SURFACE
    )
    axes = [axes] if len(live) == 1 else list(axes.flat)

    for ax, (tag, name, series) in zip(axes, live):
        ax.set_facecolor(SURFACE)
        for configuration, seed, (xs, ys) in series:
            xs, ys = smooth(xs, ys) if tag == "train/loss" else (xs, ys)
            style = STYLE.get(seed, "-")
            ax.plot(
                xs, ys, color=COLOR[configuration], linestyle=style, linewidth=1.6,
                alpha=0.95 if seed == 42 else 0.75, solid_capstyle="round",
            )
        vals = [v for _, _, (_, ys) in series for v in ys if v > 0]
        if vals and max(vals) / max(min(vals), 1e-12) > 60:
            ax.set_yscale("log")
        present = [a for a, _ in ARMS if any(x == a for x, _, _ in series)]
        n_runs = len({(a, s) for a, s, _ in series})
        if len(present) < len(ARMS):
            head = f"{name}\nonly {', '.join(SHORT[a] for a in present)} log this ({n_runs} runs)"
        else:
            head = f"{name}   (all four, {n_runs} runs)"
        ax.set_title(head, fontsize=9.5, color=INK, pad=7)
        ax.set_xlabel("training step", fontsize=8.5, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    for ax in axes[len(live):]:
        ax.set_visible(False)

    handles = [
        Line2D([], [], color=COLOR[a], linewidth=2.2, label=LABEL[a])
        for a, _ in ARMS
        if any(configuration == a for (configuration, _) in runs)
    ]
    handles += [
        Line2D([], [], color=INK2, linewidth=1.4, linestyle=st, label=f"seed {sd}")
        for sd, st, _ in SEEDS
        if any(s == sd for (_, s) in runs)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(4, len(handles)),
        frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.005),
    )
    if flat:
        subtitle += (
            "\nidentically zero in every run, so not drawn: " + ", ".join(flat)
        )
    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    fig.text(0.5, 0.958, subtitle, ha="center", va="top", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94 if flat else 0.952])
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path.name}: панелей {len(live)}"
          + (f", тождественный ноль опущен: {', '.join(flat)}" if flat else ""))


def draw_seeds(runs, path, tag="val/acceptance_decode"):
    """One panel per experiment: every seed drawn, their range shaded.

    Shares a single y-axis across panels — with free scales a flat run and a
    climbing one look alike, which is the one comparison this figure exists to
    make.
    """
    have = [a for a, _ in ARMS if any(k[0] == a and tag in runs[k] for k in runs)]
    if not have:
        print("нет данных для панели по сидам")
        return
    fig, axes = plt.subplots(
        1, len(have), figsize=(4.3 * len(have), 4.1), facecolor=SURFACE, sharey=True
    )
    axes = [axes] if len(have) == 1 else list(axes)
    lo = hi = None
    for ax, a in zip(axes, have):
        ax.set_facecolor(SURFACE)
        series = [(s, runs[(a, s)][tag]) for (c, s) in runs if c == a and tag in runs[(a, s)]]
        grid = sorted({x for _, (xs, _) in series for x in xs})
        cols = []
        for step in grid:
            vals = [ys[xs.index(step)] for _, (xs, ys) in series if step in xs]
            cols.append((step, min(vals), max(vals)) if vals else (step, None, None))
        band = [(st, a_, b_) for st, a_, b_ in cols if a_ is not None]
        if band:
            ax.fill_between([b[0] for b in band], [b[1] for b in band],
                            [b[2] for b in band], color=COLOR[a], alpha=0.18, linewidth=0)
        for seed, (xs, ys) in sorted(series):
            ax.plot(xs, ys, color=COLOR[a], linestyle=STYLE.get(seed, "-"),
                    linewidth=1.6, alpha=0.95 if seed == 42 else 0.8)
            lo = min(ys) if lo is None else min(lo, min(ys))
            hi = max(ys) if hi is None else max(hi, max(ys))
        ax.set_title(LABEL[a], fontsize=10, color=INK, pad=8)
        ax.set_xlabel("training step", fontsize=8.5, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("accepted tokens out of 31", fontsize=9, color=INK2)
    handles = [Line2D([], [], color=INK2, linewidth=1.4, linestyle=st, label=f"seed {sd}")
               for sd, st, _ in SEEDS if any(s == sd for (_, s) in runs)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Acceptance during training, every experiment and every seed",
                 fontsize=13, color=INK, y=0.98)
    fig.text(0.5, 0.915, "shaded band = seed-to-seed range · shared y-axis",
             ha="center", va="top", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0.07, 1, 0.895])
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path.name}: панелей {len(have)}")


def main():
    runs = {}
    for configuration, _ in ARMS:
        for seed, _, _ in SEEDS:
            d = run_dir(configuration, seed)
            if d:
                runs[(configuration, seed)] = load_run(d)
    if not runs:
        print("прогонов не найдено: нет checkpoints/**/lightning_logs")
        return
    print(f"прогонов: {len(runs)}")
    for (a, s) in sorted(runs, key=lambda k: (k[0], k[1])):
        print(f"  {LABEL[a]:28s} seed {s}   скаляров {len(runs[(a, s)])}")

    draw(
        LOSS_PANELS, runs, OUT / "curves_loss.png",
        "Loss curves — every logged term, every experiment, every seed",
        "colour = experiment · line style = training seed · log scale where the "
        "range demands it",
    )
    draw(
        METRIC_PANELS, runs, OUT / "curves_metrics.png",
        "Validation metrics — every logged metric, every experiment, every seed",
        "colour = experiment · line style = training seed",
    )
    draw_seeds(runs, OUT / "curves_seeds.png")


if __name__ == "__main__":
    main()
