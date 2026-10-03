"""Заменить в диффузионном виде целый слой (или только его MLP) линейным отображением.

  PYTHONPATH=. python tools/linear_block_fit_probe.py <ckpt> [layer,mlp]

Продолжение `tools/linear_attention_fit_probe.py`: внимание — около 20% весов
прохода, поэтому его замена почти ничего не экономит. Здесь заменяется то, где
сидит стоимость.

Режимы:
  layer — слой ℓ диффузионного вида целиком: h ↦ h + W·LN_ℓ(h) + b, где LN_ℓ —
          его input_layernorm; подгоняется приращение остаточного потока
          layer_ℓ(h) − h. Проход драфтера не читает ни внимание, ни MLP слоя.
  mlp   — только MLP слоя ℓ: x ↦ W x + b, x — вход MLP (после
          post_attention_layernorm); внимание слоя остаётся.

Подгонка та же: гребневая регрессия в замкнутом виде по калибровочным
состояниям, λ — по отложенной пятой части, без backward. Оценка — те же 40
состояний, что во всех пробах (базовая приёмка 3.775).

Для гибридов печатается стоимость прохода драфтера c (доля весов полного
прохода, которые он ещё читает; контекст короткий, KV не учитывается) и
ожидаемая скорость относительно Orthrus:
    ((A_k + 1) / (1 + c_k)) / ((A_0 + 1) / 2),
то есть TPF цикла «черновик + проверка» при удешевлённом черновике.
"""
import json
import os
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.getcwd())

CK = sys.argv[1]
MODES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["layer", "mlp"]
SPACING, DRAFT = 8, 31
EVAL = (["gsm8k", "humaneval"], 4, 5, 0)
CAL = (["gsm8k", "math500", "humaneval", "mbpp"], 10, 8, 10)
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)

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
cfg = T.config
mask_id = cfg.mask_token_id
n_layers, d = cfg.num_hidden_layers, cfg.hidden_size
hd = getattr(cfg, "head_dim", d // cfg.num_attention_heads)
P_ATTN = d * cfg.num_attention_heads * hd * 2 + 2 * d * cfg.num_key_value_heads * hd
P_MLP = 3 * d * cfg.intermediate_size
P_TOTAL = n_layers * (P_ATTN + P_MLP) + cfg.vocab_size * d  # выходной слой читается целиком
P_W = d * d + d
f64 = torch.float64

S = {"diff": False, "mode": None, "collect": None, "replace": set()}
MOM, SAMPLES, WB = {}, {}, {}


def reset_storage():
    MOM.clear(); SAMPLES.clear(); WB.clear()
    for L in range(n_layers):
        MOM[L] = {"xx": torch.zeros(d + 1, d + 1, dtype=f64), "xy": torch.zeros(d + 1, d, dtype=f64)}
    for part in ("hold", "eval"):
        SAMPLES[part] = {L: ([], []) for L in range(n_layers)}


def collect(L, x, y):
    if S["collect"] == "train":
        xa = torch.cat([x.cpu().double(), torch.ones(x.shape[0], 1, dtype=f64)], 1)
        MOM[L]["xx"] += xa.T @ xa
        MOM[L]["xy"] += xa.T @ y.cpu().double()
    elif S["collect"] is not None:
        xs, ys = SAMPLES[S["collect"]][L]
        xs.append(x.float().cpu())
        ys.append(y.float().cpu())


def predict(L, x):
    xa = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], 1)
    return xa @ WB[L]


def layer_hook(L):
    def hook(module, args, kwargs, output):
        if not S["diff"] or S["mode"] != "layer":
            return None
        h = args[0] if args else kwargs["hidden_states"]
        x = module.input_layernorm(h)[0]
        collect(L, x, (output - h)[0])
        if L in S["replace"]:
            return h + predict(L, x)[None]
        return None
    return hook


def mlp_hook(L):
    def hook(module, args, kwargs, output):
        if not S["diff"] or S["mode"] != "mlp":
            return None
        x = (args[0] if args else kwargs["x"])[0]
        collect(L, x, output[0])
        if L in S["replace"]:
            return predict(L, x)[None]
        return None
    return hook


for L, layer in enumerate(T.model.layers):
    layer.register_forward_hook(layer_hook(L), with_kwargs=True)
    layer.mlp.register_forward_hook(mlp_hook(L), with_kwargs=True)

eval_caches = []
with torch.no_grad():
    for ctx, _, _ in eval_states:
        c = DynamicCache(config=cfg)
        T(input_ids=ctx, position_ids=torch.arange(ctx.shape[1], device=dev)[None],
          past_key_values=c, use_cache=True)
        eval_caches.append(c)


def run(states, caches=None, collect_part=None, replace=()):
    S["collect"], S["replace"] = collect_part, set(replace)
    accs = []
    with torch.no_grad():
        for i, (ctx, anchor, target) in enumerate(states):
            if caches is not None:
                c = caches[i]
            else:
                c = DynamicCache(config=cfg)
                T(input_ids=ctx, position_ids=torch.arange(ctx.shape[1], device=dev)[None],
                  past_key_values=c, use_cache=True)
            n = ctx.shape[1]
            blk = torch.full((1, DRAFT + 1), mask_id, device=dev)
            blk[0, 0] = anchor
            S["diff"] = True
            out = T(input_ids=blk, position_ids=torch.arange(n, n + DRAFT + 1, device=dev)[None],
                    past_key_values=c, use_cache=False, is_diffusion_pass=True, ar_seq_len=n)
            S["diff"] = False
            draft = out.logits[0, :-1].argmax(-1)
            accs.append(int((draft == target).long().cumprod(0).sum()))
    S["collect"], S["replace"] = None, set()
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
    reg[-1, -1] = 0.0
    return torch.linalg.solve(xx + reg, mom["xy"])


hold_ids = set(range(4, len(cal_states), 5))
cal_train = [s for i, s in enumerate(cal_states) if i not in hold_ids]
cal_hold = [s for i, s in enumerate(cal_states) if i in hold_ids]

for mode in MODES:
    S["mode"] = mode
    reset_storage()
    run(cal_train, collect_part="train")
    run(cal_hold, collect_part="hold")
    base = run(eval_states, eval_caches, collect_part="eval")
    print(json.dumps({"режим": mode, "базовая приёмка": round(base, 3)}, ensure_ascii=False), flush=True)
    rows = []
    for L in range(n_layers):
        xh, yh = stack("hold", L)
        xe, ye = stack("eval", L)
        best = min(RIDGES, key=lambda r: sq_err(xh, yh, solve(MOM[L], r)))
        full_mom = {"xx": MOM[L]["xx"] + xh.T @ xh, "xy": MOM[L]["xy"] + xh.T @ yh}
        wb = solve(full_mom, best)
        yy = float((ye ** 2).sum())
        rel = (sq_err(xe, ye, wb) / yy) ** 0.5
        WB[L] = wb.float().to(dev)
        acc = run(eval_states, eval_caches, replace=[L])
        rows.append(acc)
        print(json.dumps({"режим": mode, "слой": L, "ridge": best, "rel_err": round(rel, 3),
                          "acc": round(acc, 3)}, ensure_ascii=False), flush=True)
    saved = (P_ATTN + P_MLP if mode == "layer" else P_MLP) - P_W
    order = sorted(range(n_layers), key=lambda L: base - rows[L])
    for k in (2, 4, 8, 12, 16, 20, 28):
        acc = run(eval_states, eval_caches, replace=order[:k])
        c = (P_TOTAL - k * saved) / P_TOTAL
        speed = ((acc + 1) / (1 + c)) / ((base + 1) / 2)
        print(json.dumps({"режим": mode, "заменено слоёв": k, "какие": sorted(order[:k]),
                          "приёмка": round(acc, 3), "стоимость черновика c": round(c, 3),
                          "скорость к Orthrus": round(speed, 3)}, ensure_ascii=False), flush=True)
