"""Аккуратный замер приёмки: фиксированный приор + несколько сидов.

Обнаружено при прогоне A/B: один и тот же чекпоинт на одних промптах дал
A_1 = 0.711 и 0.637 в двух запусках. Причина — `sample_prior` тянет новый x_0
на каждый цикл декода, то есть приёмка случайна при фиксированных весах.
Спаривание по промптам этого не ловит: оно снимает дисперсию между промптами,
а не между розыгрышами приора.

Здесь единица наблюдения — пара (промпт, сид). `decode.fixed_prior` делает
приор детерминированным внутри прогона, сид его меняет между прогонами, так что
обе оси дисперсии учтены и конфигурации спарены по обеим.

  CKPTS=name=/path/to.ckpt,other=/path2.ckpt uv run python .work/measure.py
"""
import json, math, os, sys, statistics as st

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = int(os.environ.get("PROMPTS", 16))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
MAX_NEW = int(os.environ.get("MAX_NEW", 32))
MODEL = os.environ.get("MODEL", "smollm2_135m")
BLOCK = int(os.environ.get("BLOCK", 8))
OUT = os.environ.get("OUT", "results/measure.json")
# Кривая деградации по числу шагов, одна ось для обеих моделей. Для карты
# потока это рестарты (s_k, 1); для маскирующего бейзлайна — его собственная
# абляция multi-step denoising, где используется только ДЛИНА расписания.
# n3 -- расписание, которому обучение РОВНО соответствует при
# selfcorrect_rounds=2: обучение проходит те же состояния, что декод на трёх
# прыжках. Без него замер целиком лежал на n=1, 2 и 4, то есть на состояниях,
# которых член самокоррекции не видел, и «работает ли многошаговость» проверялось
# только на переносе. n1/n2/n4 остаются: они показывают, переносится ли навык за
# обученную точку.
SCHEDULES = {
    "n1": 1,
    "n2": [[0, 1], [0.5, 1]],
    "n3": [[0, 1], [0.34, 1], [0.67, 1]],
    "n4": [[0, 1], [0.34, 1], [0.67, 1], [0.9, 1]],
}


def configuration(name, ckpt):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from src.models.factory import build_lit
    from src.eval import dataset_prompts

    with initialize_config_dir(config_dir=os.path.abspath("src/configs"), version_base=None):
        cfg = compose(config_name="eval", overrides=[
            f"checkpoint={ckpt}", f"model={MODEL}", "data=math500",
            f"decode.n_prompts={PROMPTS}", "decode.prompt_len=48",
            "decode.fixed_prior=true",
        ])
    with open_dict(cfg):
        cfg.model.backbone.device_map = None
        cfg.model.backbone.dtype = "float32"
    model = build_lit(cfg)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(dev).eval()
    prompts = [p for _, _, p in dataset_prompts(model, cfg)][:PROMPTS]
    if len(prompts) != PROMPTS:
        # Свёртка ячеек в промпты идёт по PROMPTS; если датасет отдал меньше,
        # блоки сидов перемешаются и спаривание молча станет неверным.
        raise SystemExit(f"нужно {PROMPTS} промптов, получено {len(prompts)}")

    out = {}
    for label, jumps in SCHEDULES.items():
        cells = []
        for seed in SEEDS:
            with open_dict(model.cfg):
                model.cfg.seed = seed
            for pi, ids in enumerate(prompts):
                # Глобальный RNG сеется перед КАЖДОЙ ячейкой: иначе её состояние
                # зависит от того, сколько циклов израсходовали предыдущие
                # промпты и предыдущие конфигурации, и конфигурации перестают быть спарены.
                torch.manual_seed(seed * 100003 + pi)
                try:
                    with torch.no_grad():
                        g = model.generate(input_ids=ids.to(dev), block_size=BLOCK,
                                           jumps=jumps, max_new_tokens=MAX_NEW,
                                           temperature=0.0)
                except ValueError:
                    cells = None
                    break
                acc = g["acceptance"]
                cells.append(sum(acc) / max(len(acc), 1))
            if cells is None:
                break
        if cells is None:
            continue
        passes = 1 if jumps == 1 else len(jumps)
        out[label] = {"A": sum(cells) / len(cells),
                      "tpf": (sum(cells) / len(cells) + 1) / (passes + 1),
                      # Число промптов записывается рядом с ячейками: без него
                      # график не может свернуть ячейки в независимые единицы и
                      # молча считает SE по 120 зависимым — в sqrt(сиды) уже.
                      "prompts": PROMPTS,
                      "cells": cells}
        print(f"[{name}] {label}: A={out[label]['A']:.3f}  TPF={out[label]['tpf']:.3f}  "
              f"n={len(cells)} ячеек", flush=True)
    return out


def tcrit(n):
    """Квантиль Стьюдента 0.975 с n−1 степенями свободы.

    Не 1.96 и не приближение к нему: нормальный квантиль занижает CI на 5.5%
    при n = 24 и ставит звёздочку раньше положенного, а замена его на
    1.96·(1+2/(n−1)) ошибается в другую сторону (+4.2% при n = 16).
    """
    from scipy.stats import t
    return float(t.ppf(0.975, max(n - 1, 1)))


def paired(x, y, label):
    """Единица наблюдения — ПРОМПТ, а не ячейка.

    У маскирующего драфтера декод детерминирован: приор в нём не участвует, и
    пять сидов дают побитово одинаковые ячейки. Считать SE по 120 ячейкам как
    по независимым занижало её в sqrt(5) ~ 2.24 раза и раздало звёздочки
    контрастам, которые их не заслужили. Разности усредняются внутри промпта,
    затем t по числу промптов.
    """
    raw = [b - a for a, b in zip(x, y)]
    k = len(raw) // PROMPTS
    d = [st.mean([raw[j * PROMPTS + i] for j in range(k)]) for i in range(PROMPTS)]
    n = len(d); m = sum(d) / n; se = st.stdev(d) / math.sqrt(n)
    crit = tcrit(n)
    star = "*" if abs(m) > crit * se else " "
    print(f"{label:32s} {m:+.3f}{star} t={m / se:+5.2f}  "
          f"95%CI=[{m - crit * se:+.3f},{m + crit * se:+.3f}]  n={n}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    spec = os.environ["CKPTS"]
    configurations = dict(part.split("=", 1) for part in spec.split(","))
    res = {name: configuration(name, path) for name, path in configurations.items()}
    json.dump(res, open(OUT, "w"), indent=1)
    names = list(res)
    print(f"\n=== спарено по (промпт, сид), {len(SEEDS)} сидов x {PROMPTS} промптов ===")
    for label in SCHEDULES:
        have = [n for n in names if label in res[n]]
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                paired(res[a][label]["cells"], res[b][label]["cells"],
                       f"{label}: {b} - {a}")
    for n in names:
        if "n1" in res[n] and "n2" in res[n]:
            paired(res[n]["n1"]["cells"], res[n]["n2"]["cells"], f"{n}: A_2 - A_1")
    print(f"-> {OUT}")
