"""Смешанная линеаризация диффузионного вида нашей модели: в каждом слое — что
заменить линейной подгонкой (внимание, MLP, оба, слой целиком) или ничего.

  PYTHONPATH=. python tools/mixed_linearization_probe.py <ckpt нашей модели>

Работает в нашей обвязке (Orthrus или FlowDraft из `src/models`), потому что
кампании LinearOrthrus учатся на Qwen3-0.6B: план берётся с обученной 0.6B, а не
с выпущенного 1.7B.

Компоненты слоя ℓ и их линейные замены (подгонка — гребневая регрессия в
замкнутом виде, λ на отложенных состояниях, без backward):
  attn  — выход подслоя внимания по его входу (после input_layernorm);
  mlp   — выход MLP по его входу (после post_attention_layernorm);
  layer — приращение остаточного потока h ↦ layer(h) − h по LN(h).
Состояние слоя: none | attn | mlp | attn+mlp | layer.

Наборы состояний (якорь через каждые 8 токенов жадного продолжения, 31 цель):
  fit    — подгонка W, b: 4 набора × 10 промптов со смещением 10;
  select — выбор λ и плана: каждое восьмое состояние той же калибровки,
           в подгонку не входит;
  eval   — отчёт: gsm8k и humaneval, промпты 0–3, 5 якорей — те же 40
           состояний, что во всех пробах.

Выбор плана — жадный с переоценкой: на каждом шаге из короткого списка лучших
по одиночной замене кандидатов берётся тот, что даёт наибольшую ожидаемую
скорость к Orthrus на select:
    speed = ((A + 1) / (1 + c)) / ((A₀ + 1) / 2),
c — доля весов полного прохода, которую ещё читает драфтер (выходной слой
читается всегда; KV-кэш не учтён — короткий контекст). Переоценка нужна потому,
что потери от замен складываются и одиночный рейтинг их не видит.
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
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
SHORTLIST, MAX_STEPS = 6, 40
PARTS = ("attn", "mlp", "layer")

from src.models.factory import build_lit  # noqa: E402
from src.eval import dataset_prompts  # noqa: E402

cfg_dir = os.path.abspath("src/configs")


def cfg_for(ds, n, offset):
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        return compose("eval", overrides=[f"checkpoint={CK}", f"data={ds}", f"decode.n_prompts={n}",
                                          f"decode.prompt_offset={offset}",
                                          "model.backbone.dtype=float32"])


m = build_lit(cfg_for(EVAL[0][0], EVAL[1], EVAL[3]))
m.eval()
dev = m._generation_device()


def build_states(spec):
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


eval_states = build_states(EVAL)
cal = build_states(CAL)
sel_ids = set(range(7, len(cal), 8))
fit_states = [s for i, s in enumerate(cal) if i not in sel_ids]
sel_states = [s for i, s in enumerate(cal) if i in sel_ids]
print(f"состояний: подгонка {len(fit_states)}, выбор {len(sel_states)}, отчёт {len(eval_states)}",
      flush=True)

backbone = m.orthrus.model
bcfg = backbone.config
layers = backbone.model.layers
n_layers, d = bcfg.num_hidden_layers, bcfg.hidden_size
hd = getattr(bcfg, "head_dim", d // bcfg.num_attention_heads)
COST = {"attn": d * bcfg.num_attention_heads * hd * 2 + 2 * d * bcfg.num_key_value_heads * hd,
        "mlp": 3 * d * bcfg.intermediate_size}
COST["layer"] = COST["attn"] + COST["mlp"]
P_TOTAL = n_layers * COST["layer"] + bcfg.vocab_size * d
P_W = d * d + d
f64 = torch.float64

# Флаг диффузионного прохода: все DF-вызовы идут через forward адаптера с use_df=True.
S = {"diff": False, "collect": None, "plan": {}}
_adapter_forward = m.orthrus.forward


def _flagged(*a, **k):
    prev, S["diff"] = S["diff"], bool(k.get("use_df", False))
    try:
        return _adapter_forward(*a, **k)
    finally:
        S["diff"] = prev


m.orthrus.forward = _flagged

MOM = {(L, p): {"xx": torch.zeros(d + 1, d + 1, dtype=f64), "xy": torch.zeros(d + 1, d, dtype=f64)}
       for L in range(n_layers) for p in PARTS}
SEL = {(L, p): ([], []) for L in range(n_layers) for p in PARTS}
WB = {}


def collect(key, x, y):
    if S["collect"] == "fit":
        xa = torch.cat([x.cpu().double(), torch.ones(x.shape[0], 1, dtype=f64)], 1)
        MOM[key]["xx"] += xa.T @ xa
        MOM[key]["xy"] += xa.T @ y.cpu().double()
    elif S["collect"] == "select":
        SEL[key][0].append(x.float().cpu())
        SEL[key][1].append(y.float().cpu())


def predict(key, x):
    xa = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], 1)
    return xa @ WB[key]


def replaced(L, part):
    state = S["plan"].get(L, "none")
    return state == part or (state == "attn+mlp" and part in ("attn", "mlp"))


def attn_hook(L):
    def hook(module, args, kwargs, output):
        if not S["diff"]:
            return None
        x = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])[0]
        collect((L, "attn"), x, output[0][0])
        if replaced(L, "attn"):
            return (predict((L, "attn"), x)[None],) + tuple(output[1:])
        return None
    return hook


def mlp_hook(L):
    def hook(module, args, kwargs, output):
        if not S["diff"]:
            return None
        x = (args[0] if args else kwargs["x"])[0]
        collect((L, "mlp"), x, output[0])
        if replaced(L, "mlp"):
            return predict((L, "mlp"), x)[None]
        return None
    return hook


def layer_hook(L):
    def hook(module, args, kwargs, output):
        if not S["diff"]:
            return None
        if S["collect"] is None and not replaced(L, "layer"):
            return None
        h = args[0] if args else kwargs["hidden_states"]
        out = output[0] if isinstance(output, tuple) else output
        x = module.input_layernorm(h)[0]
        collect((L, "layer"), x, (out - h)[0])
        if replaced(L, "layer"):
            new = h + predict((L, "layer"), x)[None]
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return None
    return hook


for L, layer in enumerate(layers):
    layer.self_attn.register_forward_hook(attn_hook(L), with_kwargs=True)
    layer.mlp.register_forward_hook(mlp_hook(L), with_kwargs=True)
    layer.register_forward_hook(layer_hook(L), with_kwargs=True)

times = m._jump_schedule(1)


def prefill(ctx):
    from transformers import DynamicCache
    c = DynamicCache(config=bcfg)
    m.orthrus(ctx, torch.ones_like(ctx), past_key_values=c)
    return c


eval_caches = [None] * len(eval_states)
sel_caches = [None] * len(sel_states)


def acceptance(states, caches, plan, collect_part=None):
    S["plan"], S["collect"] = dict(plan), collect_part
    accs = []
    with torch.no_grad():
        for i, (ctx, anchor, target) in enumerate(states):
            c = caches[i] if caches is not None else None
            if c is None:
                c = prefill(ctx)
                if caches is not None:
                    caches[i] = c
            torch.manual_seed(1000 + i)  # приор FlowDraft — одинаковый во всех условиях
            ids, _ = m._draft_block(c, DRAFT + 1, times, anchor_token=torch.tensor([anchor]))
            accs.append(int((ids[0] == target.to(ids.device)).long().cumprod(0).sum()))
    S["plan"], S["collect"] = {}, None
    return sum(accs) / len(accs)


def cost_of(plan):
    saved = 0
    for state in plan.values():
        if state == "layer":
            saved += COST["layer"] - P_W
        elif state == "attn+mlp":
            saved += COST["attn"] + COST["mlp"] - 2 * P_W
        elif state in ("attn", "mlp"):
            saved += COST[state] - P_W
    return (P_TOTAL - saved) / P_TOTAL


def speed(acc, c, base):
    return ((acc + 1) / (1 + c)) / ((base + 1) / 2)


# 1. Моменты для подгонки и выборки для λ.
acceptance(fit_states, None, {}, collect_part="fit")
base_sel = acceptance(sel_states, sel_caches, {}, collect_part="select")
base_eval = acceptance(eval_states, eval_caches, {})
print(json.dumps({"база": {"select": round(base_sel, 3), "eval": round(base_eval, 3)}},
                 ensure_ascii=False), flush=True)


def solve(mom, ridge):
    xx = mom["xx"]
    lam = ridge * float(torch.diagonal(xx)[:-1].mean())
    reg = torch.eye(d + 1, dtype=f64) * lam
    reg[-1, -1] = 0.0
    return torch.linalg.solve(xx + reg, mom["xy"])


for key in MOM:
    xs, ys = SEL[key]
    x = torch.cat(xs).double()
    xa = torch.cat([x, torch.ones(x.shape[0], 1, dtype=f64)], 1)
    y = torch.cat(ys).double()
    best = min(RIDGES, key=lambda r: float(((y - xa @ solve(MOM[key], r)) ** 2).sum()))
    WB[key] = solve(MOM[key], best).float().to(dev)

# 2. Одиночные замены на select.
single = {}
for L in range(n_layers):
    for part in PARTS:
        a = acceptance(sel_states, sel_caches, {L: part})
        single[(L, part)] = a
        print(json.dumps({"слой": L, "часть": part, "приёмка select": round(a, 3),
                          "потеря на 1M весов": round((base_sel - a) / ((COST[part] - P_W) / 1e6), 4)},
                         ensure_ascii=False), flush=True)


def moves(plan):
    """Допустимые шаги из плана: добавить внимание, MLP или заменить слой целиком."""
    out = []
    for L in range(n_layers):
        state = plan.get(L, "none")
        if state == "none":
            out += [(L, "attn"), (L, "mlp"), (L, "layer")]
        elif state == "attn":
            out += [(L, "attn+mlp")]
        elif state == "mlp":
            out += [(L, "attn+mlp")]
    return out


def added_part(move, plan):
    """Какой компонент добавляет шаг: для «внимание+MLP» — тот, которого не было."""
    L, part = move
    if part == "attn+mlp":
        return "mlp" if plan.get(L) == "attn" else "attn"
    return part


def score(move, plan):
    """Потеря приёмки одиночной заменой на сэкономленный вес; меньше — лучше."""
    part = added_part(move, plan)
    return (base_sel - single[(move[0], part)]) / (COST[part] - P_W)


# 3. Жадный выбор с переоценкой.
plan, path = {}, []
for step in range(MAX_STEPS):
    cands = sorted(moves(plan), key=lambda mv: score(mv, plan))[:SHORTLIST]
    if not cands:
        break
    best = None
    for L, part in cands:
        trial = dict(plan)
        trial[L] = part
        a = acceptance(sel_states, sel_caches, trial)
        c = cost_of(trial)
        sp = speed(a, c, base_sel)
        if best is None or sp > best[0]:
            best = (sp, a, c, L, part)
    sp, a, c, L, part = best
    plan[L] = part
    path.append({"шаг": step + 1, "слой": L, "часть": part, "приёмка select": round(a, 3),
                 "c": round(c, 3), "скорость select": round(sp, 3)})
    print(json.dumps(path[-1], ensure_ascii=False), flush=True)

# 4. Отчёт на eval по точкам пути.
for cut in sorted({min(k, len(path)) for k in (4, 8, 12, 16, 20, 24, 28, 32, 40)}):
    p = {}
    for row in path[:cut]:
        p[row["слой"]] = row["часть"]
    a = acceptance(eval_states, eval_caches, p)
    c = cost_of(p)
    print(json.dumps({"шагов": cut, "план": {str(k): v for k, v in sorted(p.items())},
                      "приёмка eval": round(a, 3), "c": round(c, 3),
                      "скорость eval": round(speed(a, c, base_eval), 3)}, ensure_ascii=False), flush=True)
