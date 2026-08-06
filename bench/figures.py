"""Фигуры по стенду SmolLM2-135M -> results/figures/*.png

Три вопроса, три фигуры:
  1. acceptance_vs_jumps — что делает приёмка и TPF при n = 1, 2, 4 у каждой руки
  2. training_horizon    — как контраст «многошаговое обучение минус обычное»
                           меняется с бюджетом: 2000 / 4000 / 6000 шагов
  3. contrasts           — спаренные контрасты на 6000 с доверительными
                           интервалами; всё, что пересекает ноль, незначимо

Единица наблюдения везде — пара (промпт, сид), 24 x 5 = 120 ячеек на клетку,
приор фиксирован. Спаривание снимает и разброс промптов, и разброс розыгрышей
приора; обе оси мерялись отдельно и обе велики.

  uv run python .work/figures.py
"""
import json
import math
import statistics as st
from pathlib import Path

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- палитра: референсный набор скилла, слоты в фиксированном порядке ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e7e3"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
POS, NEG, NEUTRAL = "#2a78d6", "#d03b3b", "#8b8a85"

ARMS = ["orthrus", "orthrus_ms", "fd_ms", "fd_cfm_ms", "fd_base"]
TITLES = {
    "orthrus": "orthrus",
    "orthrus_ms": "orthrus + самокоррекция",
    "fd_ms": "flowdraft + самокоррекция",
    "fd_cfm_ms": "flowdraft + самокорр. + CFM",
    "fd_base": "flowdraft (без самокорр.)",
}
COLOR = dict(zip(ARMS, SLOT))
SCHED = [("n1", 1), ("n2", 2), ("n4", 4)]


def load(name):
    # Числа читаются из results/, куда их кладёт measure.py. Раньше здесь был
    # .work/ — каталог черновиков, и фигуры зависели от того, что в нём лежит.
    p = ROOT / "results" / name
    return json.load(open(p)) if p.exists() else {}


# Каждая рука на каждом горизонте — ССЫЛКА (файл, ключ), а не слитый словарь.
# Слияние затирало значения на 2000 значениями с 6000, потому что имена рук в
# `six_k.json` те же самые, и траектория читалась как немонотонная.
HORIZONS = (2000, 4000, 6000)
WHERE = {
    "orthrus":    {2000: ("final_ab", "orthrus"),    4000: ("longer", "orthrus_4k"),
                   6000: ("six_k", "orthrus")},
    "orthrus_ms": {2000: ("final_ab", "orthrus_ms"), 4000: ("longer", "orthrus_ms_4k"),
                   6000: ("six_k", "orthrus_ms")},
    "fd_ms":      {2000: ("final_ab", "fd_ms"),      4000: ("fill", "fd_ms_4k"),
                   6000: ("six_k", "fd_ms")},
    "fd_cfm_ms":  {2000: ("final_ab", "fd_cfm_ms"),  4000: ("fill", "fd_cfm_ms_4k"),
                   6000: ("fill", "fd_cfm_ms_6k")},
    "fd_base":    {2000: ("final_ab", "fd_base"),    4000: ("fill", "fd_base_4k"),
                   6000: ("six_k", "fd_base")},
}


def cell(files, arm, steps):
    f, k = WHERE[arm][steps]
    return files.get(f, {}).get(k)


def figure_grid(files):
    """Приёмка против бюджета обучения, отдельная панель на каждое расписание.

    Три точки на кривую вместо одной: тренд по горизонту говорит больше, чем
    любое отдельное значение t, потому что межпрогонный разброс здесь порядка
    0.5 токена и одну точку он способен перевернуть.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE, sharex=True)
    for ax, (sched, n) in zip(axes, SCHED):
        style(ax)
        ends = []
        for arm in ARMS:
            ys = [(cell(files, arm, h) or {}).get(sched, {}).get("A") for h in HORIZONS]
            if any(v is None for v in ys):
                continue
            ax.plot(HORIZONS, ys, color=COLOR[arm], linewidth=1.8, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=3, clip_on=False)
            ends.append((ys[-1], arm))
        lo, hi = ax.get_ylim()
        ends.sort()
        prev = -1e9
        for ye, arm in ends:
            y = max(ye, prev + 0.055 * (hi - lo))
            prev = y
            ax.annotate(f" {ye:.2f}", xy=(HORIZONS[-1], y), xytext=(6, -3),
                        textcoords="offset points", fontsize=8.5, color=INK2,
                        annotation_clip=False)
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel("шагов обучения", fontsize=9.5, color=INK2)
        ax.set_title({1: "1 прыжок на цикл", 2: "2 прыжка на цикл",
                      4: "4 прыжка на цикл"}[n],
                     fontsize=11, color=INK, loc="left", pad=10)
        ax.set_xlim(1800, 7100)
    axes[0].set_ylabel("принято токенов из 7", fontsize=9.5, color=INK2)
    handles = [Line2D([], [], color=COLOR[a], linewidth=1.8, marker="o", markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=TITLES[a])
               for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("SmolLM2-135M · 120 ячеек на точку · math500",
                 fontsize=9.5, color=INK2, y=1.0, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(OUT / "horizon_grid.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def paired(a, b):
    """Δ = b − a по спаренным ячейкам: среднее, полуширина 95% CI, t."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    se = st.stdev(d) / math.sqrt(n)
    return m, 1.96 * se, (m / se if se else float("nan")), n


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def figure_jumps(six):
    """Приёмка и TPF против числа прыжков. Ряды — руки, значит категориально."""
    arms = [a for a in ARMS if a in six]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    x = range(len(SCHED))
    for ax, key, title, ylab in (
        (axes[0], "A", "Приёмка растёт по n только с самокоррекцией", "принято токенов из 7"),
        (axes[1], "tpf", "Пропускная способность за это платит", "токенов на форвард"),
    ):
        style(ax)
        if key == "tpf":
            # Единственная содержательная отсечка: 1.0 — скорость обычного AR.
            ax.axhline(1.0, color=INK2, linewidth=0.9, zorder=1)
            ax.annotate("скорость AR", xy=(len(SCHED) - 1, 1.0),
                        xytext=(-4, 6), textcoords="offset points",
                        ha="right", fontsize=8.5, color=INK2)
        ends = []
        for arm in arms:
            ys = [six[arm][s][key] for s, _ in SCHED]
            ax.plot(x, ys, color=COLOR[arm], linewidth=1.8, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=3, clip_on=False)
            ends.append(ys[-1])
        # Прямые подписи на концах — они же снимают предупреждение валидатора о
        # контрасте у светлых слотов. Близкие концы разводятся по вертикали,
        # иначе подписи наезжают друг на друга и перестают читаться.
        span = (max(ends) - min(ends)) or 1.0
        order = sorted(range(len(ends)), key=lambda i: ends[i])
        placed = {}
        prev = -1e9
        for i in order:
            y = max(ends[i], prev + 0.045 * span)
            placed[i] = y
            prev = y
        for i, arm in enumerate(arms):
            ax.annotate(f" {ends[i]:.2f}", xy=(x[-1], placed[i]), xytext=(7, -3),
                        textcoords="offset points", fontsize=9, color=INK2,
                        annotation_clip=False)
        ax.set_xticks(list(x), [str(n) for _, n in SCHED])
        ax.set_xlabel("прыжков на цикл", fontsize=9.5, color=INK2)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
        ax.set_xlim(-0.15, len(SCHED) - 1 + 0.45)

    handles = [Line2D([], [], color=COLOR[a], linewidth=1.8, marker="o",
                      markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                      label=TITLES[a]) for a in arms]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(arms), 3),
               frameon=False, fontsize=9.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("SmolLM2-135M, 6000 шагов, math500 · 120 ячеек на точку",
                 fontsize=9.5, color=INK2, y=1.0, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(OUT / "acceptance_vs_jumps.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_horizon(base, longer, six):
    """Контраст «с самокоррекцией минус без» против бюджета обучения.

    Это проверка гипотезы о том, что многокомпонентной цели нужен горизонт:
    одна точка не отличила бы сходящийся разрыв от шумящего.
    """
    points = [
        (2000, base.get("orthrus"), base.get("orthrus_ms")),
        (4000, longer.get("orthrus_4k"), longer.get("orthrus_ms_4k")),
        (6000, longer.get("orthrus_6k"), longer.get("orthrus_ms_6k")),
    ]
    points = [(s, a, b) for s, a, b in points if a and b]
    if len(points) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    style(ax)
    ax.axhline(0, color=INK2, linewidth=0.9, zorder=2)
    ax.annotate("нет разницы", xy=(points[-1][0], 0), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    for i, (sched, label) in enumerate((("n1", "один прыжок"), ("n4", "четыре прыжка"))):
        xs = [s for s, _, _ in points]
        ms, hs = [], []
        for _, a, b in points:
            m, h, _, _ = paired(a[sched]["cells"], b[sched]["cells"])
            ms.append(m); hs.append(h)
        ax.fill_between(xs, [m - h for m, h in zip(ms, hs)],
                        [m + h for m, h in zip(ms, hs)],
                        color=SLOT[i], alpha=0.13, linewidth=0, zorder=1)
        ax.plot(xs, ms, color=SLOT[i], linewidth=1.8, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(f"  {label}", xy=(xs[-1], ms[-1]), xytext=(6, -3),
                    textcoords="offset points", fontsize=9.5, color=INK2)
    ax.set_xticks([s for s, _, _ in points])
    ax.set_xlabel("шагов обучения", fontsize=9.5, color=INK2)
    ax.set_ylabel("Δ приёмки, самокоррекция минус без неё", fontsize=9.5, color=INK2)
    ax.set_title("Знак переворачивается с бюджетом обучения", fontsize=11,
                 color=INK, loc="left", pad=10)
    ax.annotate("полосы — 95% CI по спаренным ячейкам",
                xy=(0, -0.19), xycoords="axes fraction", fontsize=8.5, color=INK2)
    ax.set_xlim(points[0][0] - 200, points[-1][0] + 1100)
    fig.tight_layout()
    fig.savefig(OUT / "training_horizon.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_contrasts(six):
    """Спаренные контрасты на 6000 шагов. Пересёк ноль — незначим."""
    pairs = [("orthrus", "orthrus_ms"), ("orthrus", "fd_ms"),
             ("orthrus_ms", "fd_ms"), ("fd_base", "fd_ms")]
    rows = []
    for sched, n in SCHED:
        for a, b in pairs:
            if a in six and b in six:
                m, h, t, _ = paired(six[a][sched]["cells"], six[b][sched]["cells"])
                rows.append((f"n={n}", f"{TITLES[b]}  −  {TITLES[a]}", m, h, t))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(rows) + 1.9), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK2, linewidth=0.9, zorder=2)
    ys = list(range(len(rows)))[::-1]
    for y, (grp, label, m, h, t) in zip(ys, rows):
        # Диверг-пара палитры: синий — лучше, красный — хуже, серый — ноль в CI.
        c = NEUTRAL if abs(m) <= h else (POS if m > 0 else NEG)
        ax.plot([m - h, m + h], [y, y], color=c, linewidth=1.8,
                solid_capstyle="round", zorder=3)
        ax.plot([m], [y], marker="o", markersize=6.5, color=c,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        # Подпись у левого края, а не по центру: центрированная ложится
        # поперёк нулевой линии и та режет текст.
        ax.annotate(f"{grp}   {label}", xy=(0.005, y), xytext=(0, 9),
                    xycoords=("axes fraction", "data"),
                    textcoords="offset points", ha="left",
                    fontsize=9, color=INK2, annotation_clip=False)
        ax.annotate(f"{m:+.3f}   t={t:+.2f}", xy=(1.005, y), xytext=(0, -3),
                    xycoords=("axes fraction", "data"), textcoords="offset points",
                    fontsize=9, color=INK if abs(m) > h else INK2,
                    annotation_clip=False)
    ax.set_yticks([])
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("Δ приёмки (токенов), 95% CI", fontsize=9.5, color=INK2)
    ax.set_title("Спаренные контрасты на 6000 шагов", fontsize=11, color=INK,
                 loc="left", pad=14)
    ax.annotate("серый — интервал накрывает ноль, различие не установлено",
                xy=(0, -0.13 - 0.4 / len(rows)), xycoords="axes fraction",
                fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "contrasts.png", dpi=200, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)


def _scalars(arm, tag):
    """Кривая по шагам, склеенная из двух прогонов: 0->2000 и 2000->6000."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    points = []
    for root in (ROOT / "checkpoints" / "final_ab" / arm, Path("/tmp/fd-long") / arm):
        files = sorted(glob.glob(str(root / "lightning_logs" / "*" / "events*")))
        for f in files:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            try:
                ea.Reload()
            except Exception:
                continue
            if tag in ea.Tags()["scalars"]:
                points += [(e.step, e.value) for e in ea.Scalars(tag)]
    points.sort()
    return [p[0] for p in points], [p[1] for p in points]


def _smooth(xs, ys, window=25):
    if len(ys) < window:
        return xs, ys
    out = []
    for i in range(len(ys)):
        lo = max(0, i - window // 2)
        out.append(sum(ys[lo:i + 1]) / (i + 1 - lo))
    return xs, out


def figure_curves():
    """Кривые обучения: чем цель СТАНОВИТСЯ, а не чем она кончилась.

    Левая панель — общий лосс, шкала логарифмическая: у одно-компонентной цели
    он около 4, у четырёх-компонентной около 19, и на линейной оси вторая
    расплющила бы первую. Сравнивать по ней руки между собой НЕЛЬЗЯ — это
    разные функции; читается только форма каждой кривой.

    Правая панель сравнима: `loss/accepted` — точная жадная приёмка, померенная
    на каждом шаге обучения тем же он-полиси форвардом, который платит член
    самокоррекции. Это и есть та величина, которую проект оптимизирует, и до
    неё она была видна только через декод по паре валидационных промптов.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SURFACE)
    have = []
    for ax, tag, title, ylab, logy in (
        (axes[0], "train/loss", "Общий лосс (шкала логарифмическая)",
         "значение цели", True),
        (axes[1], "loss/accepted", "Приёмка на обучении, замер на каждом шаге",
         "принято токенов из 7", False),
    ):
        if logy:
            ax.set_yscale("log")
        style(ax)
        ends = []
        for arm in ARMS:
            xs, ys = _scalars(arm, tag)
            if not xs:
                continue
            xs, ys = _smooth(xs, ys)
            ax.plot(xs, ys, color=COLOR[arm], linewidth=1.6, zorder=3)
            ends.append((xs[-1], ys[-1], arm))
            if ax is axes[0]:
                have.append(arm)
        # Концы кривых сходятся близко, поэтому подписи разводятся по вертикали
        # в координатах осей — иначе две нижние наезжают друг на друга.
        lo, hi = ax.get_ylim()
        to_ax = (lambda v: (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))) \
            if logy else (lambda v: (v - lo) / (hi - lo))
        ends.sort(key=lambda e: to_ax(e[1]))
        prev = -1.0
        for xe, ye, arm in ends:
            frac = max(to_ax(ye), prev + 0.075)
            prev = frac
            ax.annotate(f" {TITLES[arm]}", xy=(xe, frac), xytext=(5, -3),
                        xycoords=("data", "axes fraction"),
                        textcoords="offset points", fontsize=8.5, color=INK2,
                        annotation_clip=False)
        ax.set_xlabel("шагов обучения", fontsize=9.5, color=INK2)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
        ax.set_xlim(0, 7900)
    axes[0].annotate("разные цели — сравнивать между собой нельзя, читается форма",
                     xy=(0, -0.21), xycoords="axes fraction", fontsize=8.5, color=INK2)
    fig.suptitle("SmolLM2-135M · скользящее среднее по 25 шагам",
                 fontsize=9.5, color=INK2, y=1.0, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(OUT / "training_curves.png", dpi=200, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    files = {f: load(f + ".json") for f in ("final_ab", "longer", "six_k", "fill")}
    figure_grid(files)
    base, longer, six = files["final_ab"], files["longer"], files["six_k"]
    if six:
        figure_jumps(six)
        figure_contrasts(six)
    figure_horizon(base, longer, six)
    figure_curves()
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.relative_to(ROOT)}  {f.stat().st_size // 1024} КБ")
