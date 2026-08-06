"""Аккуратный замер приёмки: фиксированный приор + несколько сидов.

Обнаружено при прогоне A/B: один и тот же чекпоинт на одних промптах дал
A_1 = 0.711 и 0.637 в двух запусках. Причина — `sample_prior` тянет новый x_0
на каждый цикл декода, то есть приёмка случайна при фиксированных весах.
Спаривание по промптам этого не ловит: оно снимает дисперсию между промптами,
а не между розыгрышами приора.

Здесь единица наблюдения — пара (промпт, сид). `decode.fixed_prior` делает
приор детерминированным внутри прогона, сид его меняет между прогонами, так что
обе оси дисперсии учтены и руки спарены по обеим.

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
OUT = os.environ.get("OUT", ".work/measure.json")
# Кривая деградации по числу шагов, одна ось для обеих моделей. Для карты
# потока это рестарты (s_k, 1); для маскирующего бейзлайна — его собственная
# абляция multi-step denoising, где используется только ДЛИНА расписания.
SCHEDULES = {
    "n1": 1,
    "n2": [[0, 1], [0.5, 1]],
    "n4": [[0, 1], [0.34, 1], [0.67, 1], [0.9, 1]],
}


def arm(name, ckpt):
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

    out = {}
    for label, jumps in SCHEDULES.items():
        cells = []
        for seed in SEEDS:
            with open_dict(model.cfg):
                model.cfg.seed = seed
            for ids in prompts:
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
        legs = 1 if jumps == 1 else len(jumps)
        out[label] = {"A": sum(cells) / len(cells),
                      "tpf": (sum(cells) / len(cells) + 1) / (legs + 1),
                      "cells": cells}
        print(f"[{name}] {label}: A={out[label]['A']:.3f}  TPF={out[label]['tpf']:.3f}  "
              f"n={len(cells)} ячеек", flush=True)
    return out


def paired(x, y, label):
    d = [b - a for a, b in zip(x, y)]
    n = len(d); m = sum(d) / n; se = st.stdev(d) / math.sqrt(n)
    star = "*" if abs(m) > 1.96 * se else " "
    print(f"{label:32s} {m:+.3f}{star} t={m / se:+5.2f}  "
          f"95%CI=[{m - 1.96 * se:+.3f},{m + 1.96 * se:+.3f}]  n={n}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    spec = os.environ["CKPTS"]
    arms = dict(part.split("=", 1) for part in spec.split(","))
    res = {name: arm(name, path) for name, path in arms.items()}
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
