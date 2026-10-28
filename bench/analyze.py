"""Разбор строгого замера: интервалы, значимость, дисперсионный анализ.

    uv run python bench/analyze.py [--data /tmp/fd-measure] [--step 0020000]

Читает попромптовые строки из `pp-*.jsonl`, которые пишет `src/eval.py` при
заданном `per_prompt_file`. Работает и на частичных данных — считает по тому,
что уже записано, и честно сообщает, сколько сочетаний недостаёт.

--------------------------------------------------------------------------
ГЛАВНОЕ ОГРАНИЧЕНИЕ, которое не чинится никакой статистикой ниже

Сид обучения ОДИН. Всё, что здесь считается, описывает разброс по промптам, то
есть шум ИЗМЕРЕНИЯ на данном прогоне. Это не разброс МЕТОДА между прогонами.
Интервал вида "конфигурация A выше конфигурации B на 0.43 +/- 0.05" означает "на этих промптах
эта обученная карта лучше той", а не "этот способ обучения лучше того". Чтобы
второе, нужен второй сид обучения и анализ с сидом как случайным фактором; при
двух сидах это df = 1 и t(0.975) = 12.71, то есть почти любой эффект окажется
незначимым. Проект на этом уже обжигался.

--------------------------------------------------------------------------
ЕДИНИЦА НАБЛЮДЕНИЯ И ПАРНОСТЬ

Единица — ПРОМПТ. Одни и те же промпты подаются всем рукам, поэтому контрасты
считаются попромптовыми разностями (парный дизайн), а не разностью средних:
парность снимает разброс промптов, который здесь доминирует.

Считать ячейки (промпт x что-нибудь ещё) независимыми нельзя. В этом проекте
такая ошибка уже случалась: SE считалась по 120 ячейкам при 24 независимых
промптах и была занижена в sqrt(5).

--------------------------------------------------------------------------
ЧТО СЧИТАЕТСЯ

1. Средняя приёмка и TPF по рукам с интервалом Стьюдента (не 1.96).
2. Парные контрасты между конфигурациями: t по попромптовым разностям, поправка
   Холма на множественность внутри семейства сравнений.
3. Дисперсионный анализ с повторными измерениями: конфигурация x расписание x горизонт,
   субъект — промпт. Поправка Гринхауса-Гейссера на нарушение сферичности.
4. Разбивка по датасетам: держится ли эффект на всех шести или тянется одним.
5. Перекрёстная проверка: промпты делятся на блоки, ранжирование конфигураций считается
   в каждом, устойчивость меряется тау Кендалла между блоками.
"""
import argparse
import glob
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats


def load(root):
    rows = []
    # Канонические замеры лежат в data/measurements/ и версионируются; шаблон
    # pp-*.jsonl остаётся для свежих прогонов eval.py в рабочий каталог.
    paths = sorted(glob.glob(os.path.join(root, "pp-*.jsonl"))
                   + glob.glob(os.path.join(root, "*per-prompt.jsonl")))
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Оборванная строка на хвосте файла: рабочего могли убить в
                # середине записи. Пропускаем её, а не роняем весь разбор.
                continue
    return rows


def key_of(row):
    """(конфигурация, шаг, датасет, расписание) — из полей самой строки.

    Отдельная метка не нужна и была бы лишней точкой отказа: `src/eval.py` уже
    пишет `checkpoint`, `checkpoint_step`, `dataset` и `jumps`, и этого хватает.
    Расписание сводится к числу шагов уточнения, потому что именно оно и есть ось.
    """
    ck = row.get("checkpoint")
    if not ck:
        return None
    parts = str(ck).replace("\\", "/").split("/")
    configuration = parts[-2] if len(parts) >= 2 else "?"
    step = f"{int(row.get('checkpoint_step', 0)):07d}"
    ds = str(row.get("dataset", "?"))
    j = row.get("jumps")
    passes = len(j) if isinstance(j, (list, tuple)) else int(j or 1)
    return (configuration, step, ds, f"n{passes}")


def tcrit(n, alpha=0.05):
    return float(stats.t.ppf(1 - alpha / 2, max(n - 1, 1)))


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return (float(v.mean()) if n else float("nan")), float("nan"), n
    m = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n))
    return m, tcrit(n) * se, n


def paired(a_map, b_map):
    """Разности по ОБЩИМ промптам. Возвращает (среднее, полуширина, t, p, n)."""
    common = sorted(set(a_map) & set(b_map))
    if len(common) < 2:
        return None
    d = np.array([a_map[k] - b_map[k] for k in common], dtype=float)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    if se == 0.0:
        return m, 0.0, float("inf") if m else 0.0, 0.0 if m else 1.0, n
    t = m / se
    p = float(2 * stats.t.sf(abs(t), n - 1))
    return m, tcrit(n) * se, t, p, n


def bootstrap_paired(a_map, b_map, n_boot=10000, seed=0):
    """Стратифицированный парный бутстрэп разности по промптам.

    Пересэмплируются ПРОМПТЫ с возвращением, внутри каждого датасета отдельно:
    без стратификации доля датасетов гуляет от выборки к выборке, и интервал
    раздувается составом, а не эффектом.

    Отвечает на вопрос «устоит ли разница на ДРУГОЙ выборке промптов из того же
    распределения». Не отвечает на вопрос «устоит ли она при другом прогоне
    обучения»: обучающая случайность здесь не пересэмплируется вовсе.

    Интервал перцентильный. BCa не берётся сознательно: он поправляет смещение
    и асимметрию оценки, а среднее разностей несмещено, и при n порядка сотен
    поправка меньше ширины самого интервала.
    """
    common = sorted(set(a_map) & set(b_map))
    if len(common) < 8:
        return None
    by_ds = defaultdict(list)
    for k in common:
        by_ds[k.split(":")[0] if ":" in k else "all"].append(k)
    diffs = {k: a_map[k] - b_map[k] for k in common}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    groups = [np.array([diffs[k] for k in ks], dtype=float) for ks in by_ds.values()]
    sizes = [len(g) for g in groups]
    total = sum(sizes)
    for b in range(n_boot):
        acc = 0.0
        for g, m in zip(groups, sizes):
            acc += g[rng.integers(0, m, m)].sum()
        means[b] = acc / total
    lo, hi = np.percentile(means, [2.5, 97.5])
    point = float(np.mean(list(diffs.values())))
    # доля бутстрэп-выборок по ту сторону нуля -- односторонняя оценка
    frac = float((means > 0).mean())
    return point, float(lo), float(hi), frac, len(common)


def sign_test_blocks(a_map, b_map, n_blocks=20):
    """Критерий знаков по блокам промптов.

    Самый робастный из трёх: не предполагает ни нормальности, ни симметрии,
    читает только знак. Платит за это мощностью — величина разности
    отбрасывается целиком.
    """
    common = sorted(set(a_map) & set(b_map))
    if len(common) < n_blocks * 2:
        n_blocks = max(4, len(common) // 4)
    if len(common) < 8:
        return None
    wins = ties = 0
    for b in range(n_blocks):
        ks = [k for i, k in enumerate(common) if i % n_blocks == b]
        if not ks:
            continue
        d = float(np.mean([a_map[k] - b_map[k] for k in ks]))
        if d > 0:
            wins += 1
        elif d == 0:
            ties += 1
    n = n_blocks - ties
    if n < 1:
        return None
    p = float(stats.binomtest(wins, n, 0.5).pvalue)
    return wins, n, p


def wilcoxon_datasets(per_dataset_diffs):
    """Критерий знаковых рангов Вилкоксона по ДАТАСЕТАМ.

    Единица наблюдения здесь — задача, а не промпт. Отвечает на другой вопрос:
    систематически ли одна конфигурация лучше другой ПО РАЗНЫМ ЗАДАЧАМ, а не по разным
    промптам одной задачи. Это более сильное утверждение о переносимости.

    Цена — крошечное n. При шести датасетах минимально достижимое двустороннее
    p равно 2/2^6 = 0.031, и оно достигается, только если все шесть разностей
    одного знака. При пяти датасетах минимум уже 0.0625, то есть значимости на
    уровне 0.05 не бывает в принципе.
    """
    d = [x for x in per_dataset_diffs if x is not None]
    if len(d) < 5:
        return None
    try:
        res = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return float(np.mean(d)), int(len(d)), float(res.statistic), float(res.pvalue)
    except ValueError:
        return None


def holm(pairs):
    """Поправка Холма: строже Бонферрони не бывает, но мощнее его."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pairs[idx][1])
        running = max(running, val)
        adj[idx] = running
    return adj


def rm_anova(cells, factor_names):
    """Однофакторный дисперсионный анализ с повторными измерениями.

    `cells`: dict уровень -> dict промпт -> значение. Субъект — промпт, каждый
    уровень измерен на каждом субъекте. Считается F, затем поправка
    Гринхауса-Гейссера: сферичность в таких данных не выполняется почти
    никогда, и без поправки p занижен.
    """
    levels = sorted(cells)
    if len(levels) < 2:
        return None
    subjects = sorted(set.intersection(*[set(cells[l]) for l in levels]))
    if len(subjects) < 3:
        return None
    y = np.array([[cells[l][s] for l in levels] for s in subjects], dtype=float)
    n, k = y.shape
    grand = y.mean()
    ss_subj = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_cond = n * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_tot = ((y - grand) ** 2).sum()
    ss_err = ss_tot - ss_subj - ss_cond
    df_cond, df_err = k - 1, (n - 1) * (k - 1)
    if df_err <= 0 or ss_err <= 0:
        return None
    f = (ss_cond / df_cond) / (ss_err / df_err)
    # Гринхаус-Гейссер
    cov = np.cov(y, rowvar=False)
    cm = cov - cov.mean(axis=0) - cov.mean(axis=1)[:, None] + cov.mean()
    eps = (np.trace(cm) ** 2) / (k - 1 if k > 1 else 1) / max((cm ** 2).sum(), 1e-30)
    eps = float(min(1.0, max(eps, 1.0 / (k - 1))))
    p = float(stats.f.sf(f, df_cond * eps, df_err * eps))
    eta = ss_cond / (ss_cond + ss_err)
    return dict(F=float(f), p=p, eps=eps, df=(df_cond, df_err), eta2=float(eta),
                n_subjects=n, n_levels=k, factor=factor_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/measurements")
    ap.add_argument("--step", default="0020000", help="снимок для основных таблиц")
    ap.add_argument("--sched", default="n3", help="расписание для основных таблиц")
    ap.add_argument("--metric", default="acceptance", choices=["acceptance", "tpf"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42,
                    help="сид обучения для основных таблиц")
    args = ap.parse_args()

    rows = load(args.data)
    seeds_present = {r.get("training_seed") for r in rows if r.get("training_seed") is not None}
    if seeds_present:
        rows = [r for r in rows if r.get("training_seed") in (None, args.seed)]
    if not rows:
        print(f"нет данных в {args.data}/pp-*.jsonl")
        return
    # (конфигурация, шаг, датасет, расписание) -> {промпт: значение}
    cube = defaultdict(dict)
    for r in rows:
        k = key_of(r)
        if k is None or r.get(args.metric) is None:
            continue
        configuration, step, ds, sc = k
        cube[(configuration, step, ds, sc)][f"{ds}:{r['prompt_index']}"] = float(r[args.metric])

    configurations = sorted({k[0] for k in cube})
    steps = sorted({k[1] for k in cube})
    dsets = sorted({k[2] for k in cube})
    scheds = sorted({k[3] for k in cube})
    print(f"ЗАГРУЖЕНО: {len(rows)} строк, {len(cube)} сочетаний")
    print(f"  конфигураций {len(configurations)}, снимков {len(steps)}, датасетов {len(dsets)}, "
          f"расписаний {len(scheds)}")
    full = len(configurations) * len(steps) * len(dsets) * len(scheds)
    if len(cube) < full:
        print(f"  ЧАСТИЧНЫЕ ДАННЫЕ: {len(cube)} из {full} ожидаемых сочетаний")
    print()
    if len(seeds_present) > 1:
        print(f"!! В данных сиды {sorted(seeds_present)}; таблицы ниже считаны по "
              f"сиду {args.seed}. Ключ ячейки — (конфигурация, снимок, датасет,")
        print("   расписание) БЕЗ сида, поэтому смешивать их здесь нельзя: строки")
        print("   разных сидов затирают друг друга. Разброс МЕТОДА по сидам — в")
        print("   EXPERIMENTS.md §4, там единица наблюдения сид, а не промпт. !!")
    else:
        print(f"!! Сид обучения ОДИН ({sorted(seeds_present) or '?'}). Всё ниже — "
              "шум ИЗМЕРЕНИЯ, не разброс метода. !!")
    print()

    def pool(configuration, step=None, sched=None, ds=None):
        """Объединить промпты по неуказанным осям."""
        out = {}
        for (a, st, d, sc), vals in cube.items():
            if a != configuration:
                continue
            if step and st != step:
                continue
            if sched and sc != sched:
                continue
            if ds and d != ds:
                continue
            for p, v in vals.items():
                out[f"{st}|{sc}|{p}"] = v
        return out

    # --- 1. средние с интервалами -----------------------------------------
    print(f"=== 1. {args.metric} на снимке {args.step}, расписание {args.sched} ===")
    print(f"  {'конфигурация':40s} {'среднее':>9} {'95% CI':>16} {'n':>5}")
    means = {}
    for configuration in configurations:
        vals = list(pool(configuration, args.step, args.sched).values())
        if not vals:
            continue
        m, h, n = mean_ci(vals)
        means[configuration] = m
        print(f"  {configuration:40s} {m:9.3f} {'[%.3f, %.3f]' % (m - h, m + h):>16} {n:5d}")

    # --- 2. парные контрасты ----------------------------------------------
    print(f"\n=== 2. Парные контрасты (парный t по промптам, поправка Холма) ===")
    tests = []
    for a, b in itertools.combinations(configurations, 2):
        ma, mb = pool(a, args.step, args.sched), pool(b, args.step, args.sched)
        res = paired(ma, mb)
        if res:
            tests.append(((a, b), res[3], res))
    if tests:
        adj = holm([(t[0], t[1]) for t in tests])
        tests_sorted = sorted(range(len(tests)), key=lambda i: -abs(tests[i][2][0]))
        print(f"  {'контраст':58s} {'Δ':>8} {'95% CI':>16} {'t':>7} {'p Холма':>9}")
        for i in tests_sorted:
            (a, b), _, (m, h, t, p, n) = tests[i]
            star = "*" if adj[i] < 0.05 else " "
            lbl = f"{a.replace('smollm_','')} − {b.replace('smollm_','')}"
            print(f"  {lbl:58s} {m:+8.3f} {'[%+.3f,%+.3f]' % (m-h, m+h):>16} "
                  f"{t:+7.2f} {adj[i]:9.4f}{star}")

    # --- 3. дисперсионный анализ ------------------------------------------
    print(f"\n=== 3. Дисперсионный анализ с повторными измерениями ===")
    for name, axis in (("конфигурация", 0), ("расписание", 3), ("горизонт", 1)):
        cells = defaultdict(dict)
        for (a, st, d, sc), vals in cube.items():
            if axis == 0 and (st != args.step or sc != args.sched):
                continue
            if axis == 3 and (a != "smollm_flowdraft_multistep" or st != args.step):
                continue
            if axis == 1 and (a != "smollm_flowdraft_multistep" or sc != args.sched):
                continue
            lvl = (a, st, d, sc)[axis]
            for p, v in vals.items():
                cells[lvl][f"{d}:{p}"] = v
        res = rm_anova(cells, name)
        if res:
            print(f"  фактор {name:12s} F({res['df'][0]},{res['df'][1]}) = {res['F']:8.2f}"
                  f"  p = {res['p']:.3e}  eta^2 = {res['eta2']:.3f}"
                  f"  ГГ eps = {res['eps']:.3f}  уровней {res['n_levels']}, "
                  f"промптов {res['n_subjects']}")
        else:
            print(f"  фактор {name:12s} данных недостаточно")

    # --- 4. по датасетам ---------------------------------------------------
    print(f"\n=== 4. Ключевые контрасты по датасетам ===")
    key_pairs = [("smollm_flowdraft_multistep", "smollm_flowdraft", "самокоррекция, симплекс"),
                 ("smollm_orthrus_multistep", "smollm_masked_baseline", "самокоррекция, маска"),
                 ("smollm_flow_selfcorrect_gate", "smollm_flowdraft_multistep", "гейт"),
                 ("smollm_flowdraft_multistep", "smollm_orthrus", "итог против Orthrus")]
    print(f"  {'контраст':28s} " + " ".join(f"{d[:8]:>9}" for d in dsets))
    for a, b, lbl in key_pairs:
        cells = []
        for d in dsets:
            res = paired(pool(a, args.step, args.sched, d), pool(b, args.step, args.sched, d))
            cells.append(f"{res[0]:+9.2f}" if res else "        -")
        print(f"  {lbl:28s} " + " ".join(cells))

    # --- 4b. три метода повторной выборки ---------------------------------
    print(f"\n=== 4b. Повторная выборка: бутстрэп, знаки, Вилкоксон ===")
    print("  бутстрэп и знаки -- вопрос «устоит ли на ДРУГИХ ПРОМПТАХ»;")
    print("  Вилкоксон по датасетам -- вопрос «систематично ли ПО ЗАДАЧАМ».")
    print("  Ни один не отвечает на вопрос «устоит ли при другом прогоне обучения»:")
    print("  обучающая случайность здесь не пересэмплируется вовсе.")
    print()
    print(f"  {'контраст':30s} {'Δ':>7} {'бутстрэп 95%':>18} {'знаки':>12} {'Вилкоксон':>16}")
    for a, b, lbl in key_pairs:
        ma, mb = pool(a, args.step, args.sched), pool(b, args.step, args.sched)
        bs = bootstrap_paired(ma, mb)
        sg = sign_test_blocks(ma, mb)
        per_ds = []
        for d in dsets:
            r = paired(pool(a, args.step, args.sched, d), pool(b, args.step, args.sched, d))
            per_ds.append(r[0] if r else None)
        wx = wilcoxon_datasets(per_ds)
        c_bs = f"[{bs[1]:+.3f},{bs[2]:+.3f}]" if bs else "-"
        c_sg = f"{sg[0]}/{sg[1]} p={sg[2]:.3f}" if sg else "-"
        c_wx = f"n={wx[1]} p={wx[3]:.4f}" if wx else "мало датасетов"
        delta = f"{bs[0]:+7.3f}" if bs else "      -"
        print(f"  {lbl:30s} {delta} {c_bs:>18} {c_sg:>12} {c_wx:>16}")

    # --- 5. перекрёстная проверка -----------------------------------------
    print(f"\n=== 5. Устойчивость ранжирования, {args.folds} блоков промптов ===")
    ranks = []
    for f in range(args.folds):
        scores = {}
        for configuration in configurations:
            vals = [v for i, (k, v) in enumerate(sorted(pool(configuration, args.step, args.sched).items()))
                    if i % args.folds == f]
            if vals:
                scores[configuration] = float(np.mean(vals))
        if len(scores) == len(configurations):
            ranks.append([a for a, _ in sorted(scores.items(), key=lambda x: -x[1])])
    if len(ranks) >= 2:
        taus = []
        for i, j in itertools.combinations(range(len(ranks)), 2):
            xi = [ranks[i].index(a) for a in configurations]
            xj = [ranks[j].index(a) for a in configurations]
            taus.append(stats.kendalltau(xi, xj).statistic)
        print(f"  тау Кендалла между блоками: среднее {np.mean(taus):.3f}, "
              f"минимум {np.min(taus):.3f}")
        print(f"  победитель по блокам: {[r[0].replace('smollm_','') for r in ranks]}")
    else:
        print("  блоков недостаточно")


if __name__ == "__main__":
    main()
