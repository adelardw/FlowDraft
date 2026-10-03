"""Заменить диффузионное внимание слоя обученным линейным слоем Wx + b.

  PYTHONPATH=. python tools/linear_attention_fit_probe.py <ckpt>

<ckpt> — выпущенный Orthrus, переложенный `tools/import_orthrus.py`; нужен
только для промптов и жадных продолжений нашей обвязкой. Замер — на модели
авторов (`chiennv/Orthrus-Qwen3-1.7B`, их код).

Цель для слоя ℓ:
    min_{W,b}  Σ ‖ A_ℓ(x) − (W x + b) ‖²,
x — вход подслоя внимания (после input_layernorm) в диффузионном проходе,
A_ℓ(x) — его выход после o_proj_diff, то есть ровно то, что прибавляется к
остаточному потоку. Остальная модель заморожена, поэтому это гребневая
регрессия с решением в замкнутом виде:
    [W b] = (XᵀX + λI)⁻¹ XᵀY.
Нужны только моменты XᵀX и XᵀY, накопленные проходами вперёд; backward не
нужен вовсе, память — O(d²) на слой.

Данные:
  калибровка — gsm8k, math500, humaneval, mbpp, промпты со смещением 10;
               каждое пятое состояние отложено для выбора λ;
  оценка     — те же 40 состояний, что у `taylor_attention_probe.py` и
               `twin_copy_probe.py` (gsm8k и humaneval, промпты 0–3, 5 якорей).

Печатает по каждому слою:
  rel_err   — √(Σ‖ŷ − y‖² / Σ‖y‖²) на оценочных состояниях для Wx + b;
  rel_err_b — то же для одной константы b (среднее выхода): сколько даёт W;
  acc_lin   — приёмка, если в ЭТОМ слое внимание заменено на Wx + b.
Затем гибрид: замена в k слоях, где она стоит меньше всего.
"""
import json
import os
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.getcwd())

CK = sys.argv[1]
SPACING, DRAFT = 8, 31
EVAL = (["gsm8k", "humaneval"], 4, 5, 0)
CAL = (["gsm8k", "math500", "humaneval", "mbpp"], 10, 8, 10)
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1)

from src.models.factory import build_lit  # noqa: E402
from src.eval import dataset_prompts  # noqa: E402

cfg_dir = os.path.abspath("src/configs")


def cfg_for(ds, n, offset):
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        return compose("eval", overrides=[f"checkpoint={CK}", f"data={ds}", f"decode.n_prompts={n}",
                                          f"decode.prompt_offset={offset}",
                                          "model.backbone.dtype=float32"])


def build_states(m, spec):
    datasets, n, p, offset = spec
    out = []
    for ds in datasets:
        for _, _, ids in dataset_prompts(m, cfg_for(ds, n, offset)):
            ar = m.ar_generate(input_ids=ids, max_new_tokens=SPACING * (p - 1) + 1 + DRAFT,
                               eos_token_id=m.tokenizer.eos_token_id)
            full = torch.cat([ids.to(dev), torch.tensor([ar["new_tokens"]], device=dev)], 1)
            for j in range(p):
                t = ids.shape[1] + SPACING * j
                if t + DRAFT >= full.shape[1]:
                    break
                out.append((full[:, :t], full[0, t].item(), full[0, t + 1: t + 1 + DRAFT]))
    return out


m = build_lit(cfg_for(EVAL[0][0], EVAL[1], EVAL[3]))
dev = m._generation_device()
eval_states = build_states(m, EVAL)
cal_states = build_states(m, CAL)
del m
print(f"состояний: оценка {len(eval_states)}, калибровка {len(cal_states)}", flush=True)

from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

T = AutoModelForCausalLM.from_pretrained("chiennv/Orthrus-Qwen3-1.7B", dtype=torch.float32,
                                         trust_remote_code=True, attn_implementation="sdpa")
T = T.to(dev).eval()
mask_id = T.config.mask_token_id
n_layers, d = T.config.num_hidden_layers, T.config.hidden_size
f64 = torch.float64


# Обучающая часть копит моменты в float64; отложенная и оценочная хранят сами
# выборки в float32 — их мало, а моменты на 28 слоёв заняли бы гигабайты.
MOM = {L: {"xx": torch.zeros(d + 1, d + 1, dtype=f64), "xy": torch.zeros(d + 1, d, dtype=f64)}
       for L in range(n_layers)}
SAMPLES = {"hold": {L: ([], []) for L in range(n_layers)},
           "eval": {L: ([], []) for L in range(n_layers)}}
MODE = {"collect": None, "replace": set()}
WB = {}


def make_hook(L):
    def hook(module, args, kwargs, output):
        if not kwargs.get("is_diffusion_pass"):
            return None
        x = kwargs["hidden_states"][0]
        if MODE["collect"] == "train":
            xa = torch.cat([x.cpu().double(), torch.ones(x.shape[0], 1, dtype=f64)], 1)
            MOM[L]["xx"] += xa.T @ xa
            MOM[L]["xy"] += xa.T @ output[0][0].cpu().double()
        elif MODE["collect"] is not None:
            xs, ys = SAMPLES[MODE["collect"]][L]
            xs.append(x.float().cpu())
            ys.append(output[0][0].float().cpu())
        if L in MODE["replace"]:
            xa = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], 1)
            return ((xa @ WB[L].to(x.device, x.dtype))[None], output[1])
        return None
    return hook


for L, layer in enumerate(T.model.layers):
    layer.self_attn.register_forward_hook(make_hook(L), with_kwargs=True)


def run(states, collect=None, replace=(), keep_caches=None):
    MODE["collect"], MODE["replace"] = collect, set(replace)
    accs = []
    with torch.no_grad():
        for i, (ctx, anchor, target) in enumerate(states):
            if keep_caches is not None and i < len(keep_caches):
                c = keep_caches[i]
            else:
                c = DynamicCache(config=T.config)
                MODE_c, MODE["collect"] = MODE["collect"], None
                T(input_ids=ctx, position_ids=torch.arange(ctx.shape[1], device=dev)[None],
                  past_key_values=c, use_cache=True)
                MODE["collect"] = MODE_c
                if keep_caches is not None:
                    keep_caches.append(c)
            Lc = ctx.shape[1]
            blk = torch.full((1, DRAFT + 1), mask_id, device=dev)
            blk[0, 0] = anchor
            out = T(input_ids=blk, position_ids=torch.arange(Lc, Lc + DRAFT + 1, device=dev)[None],
                    past_key_values=c, use_cache=False, is_diffusion_pass=True, ar_seq_len=Lc)
            draft = out.logits[0, :-1].argmax(-1)
            accs.append(int((draft == target).long().cumprod(0).sum()))
    MODE["collect"], MODE["replace"] = None, set()
    return sum(accs) / len(accs)


def stack(part, L):
    xs, ys = SAMPLES[part][L]
    x = torch.cat(xs).double()
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=f64)], 1), torch.cat(ys).double()


def sq_err(xa, y, wb):
    return float(((y - xa @ wb) ** 2).sum())


def solve(mom, ridge):
    xx = mom["xx"]
    lam = ridge * float(torch.diagonal(xx)[:-1].mean())
    reg = torch.eye(d + 1, dtype=f64) * lam
    reg[-1, -1] = 0.0  # смещение не штрафуем
    return torch.linalg.solve(xx + reg, mom["xy"])


hold_ids = set(range(4, len(cal_states), 5))
run([s for i, s in enumerate(cal_states) if i not in hold_ids], collect="train")
run([s for i, s in enumerate(cal_states) if i in hold_ids], collect="hold")
eval_caches = []
base = run(eval_states, collect="eval", keep_caches=eval_caches)
print(json.dumps({"базовая приёмка": round(base, 3)}, ensure_ascii=False), flush=True)

rows = []
for L in range(n_layers):
    xh, yh = stack("hold", L)
    xe, ye = stack("eval", L)
    best = min(RIDGES, key=lambda r: sq_err(xh, yh, solve(MOM[L], r)))
    # Окончательная подгонка — на всей калибровке при выбранном λ.
    full_mom = {"xx": MOM[L]["xx"] + xh.T @ xh, "xy": MOM[L]["xy"] + xh.T @ yh}
    wb = solve(full_mom, best)
    wb_b = torch.zeros(d + 1, d, dtype=f64)
    wb_b[-1] = full_mom["xy"][-1] / full_mom["xx"][-1, -1]
    yy = float((ye ** 2).sum())
    rel = (sq_err(xe, ye, wb) / yy) ** 0.5
    rel_b = (sq_err(xe, ye, wb_b) / yy) ** 0.5
    WB[L] = wb.float().to(dev)
    acc = run(eval_states, replace=[L], keep_caches=eval_caches)
    rows.append(acc)
    print(json.dumps({"слой": L, "ridge": best, "rel_err": round(rel, 3), "rel_err_b": round(rel_b, 3),
                      "acc_lin": round(acc, 3)}, ensure_ascii=False), flush=True)

order = sorted(range(n_layers), key=lambda L: base - rows[L])
for k in (2, 4, 8, 12, 16, 28):
    print(json.dumps({"заменено слоёв": k, "какие": sorted(order[:k]),
                      "приёмка": round(run(eval_states, replace=order[:k], keep_caches=eval_caches), 3)},
                     ensure_ascii=False), flush=True)
