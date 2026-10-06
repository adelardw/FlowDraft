"""Проверка идеи: линеаризовать внимание диффузионного вида рядом Тейлора.

  PYTHONPATH=. python tools/taylor_attention_probe.py <ckpt> <наборы через запятую> <промптов> <якорей> [порядок]

<ckpt> — выпущенный Orthrus, переложенный `tools/import_orthrus.py`; он нужен
только для промптов и жадного продолжения нашей обвязкой. Сам замер идёт на
модели авторов (`chiennv/Orthrus-Qwen3-1.7B`, их код), в диффузионном проходе.

Ряд Тейлора первого порядка для softmax по ключам j при оценках S_ij:
    softmax(S)_ij ≈ (1 + S_ij − S̄_i) / n,   S̄_i — среднее по j,
    o_i ≈ v̄ + (1/n) Σ_j (S_ij − S̄_i) v_j.
Годится, только если |S_ij − S̄_i| ≪ 1 для всех j. Ряд второго порядка
(порядок = 2) берёт e^δ ≈ 1 + δ + δ²/2 и нормирует веса на их сумму; он
положителен при любом δ, а пренебрегает лишь членами от δ³.

По каждому слою:
  spread   — среднее по запросам стандартное отклонение S_ij − S̄_i;
  maxdev   — среднее по запросам max_j (S_ij − S̄_i);
  sink     — доля внимания на первый токен;
  e2, e1, e0 — относительная ошибка выхода внимания для ряда второго, первого
             и нулевого порядка (нулевой — равномерное внимание, o_i = v̄);
  small    — доля ключей с |δ| ≤ 1.5, где отброшенный член δ³/6 ещё мал;
  mass_out — доля массы внимания на ключах с |δ| > 1.5, то есть там, где ряд
             заведомо неверен;
  acc_*    — средняя приёмка, если в ЭТОМ слое внимание заменить рядом
             первого порядка (taylor) или выкинуть (drop).
Приёмка состояния = длина совпадающего префикса argmax черновика с жадным
AR-продолжением, то есть ровно то, что примет жадная проверка.
"""
import json
import math
import os
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.getcwd())

CK, DATASETS, N, P = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4])
ORDER = int(sys.argv[5]) if len(sys.argv) > 5 else 1
SPACING, DRAFT = 8, 31

# 1. Промпты и жадные продолжения — нашей обвязкой, как во всех замерах.
from src.models.factory import build_lit  # noqa: E402
from src.eval import dataset_prompts  # noqa: E402

cfg_dir = os.path.abspath("src/configs")


def cfg_for(ds):
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        return compose("eval", overrides=[f"checkpoint={CK}", f"data={ds}",
                                          f"decode.n_prompts={N}", "model.backbone.dtype=float32"])


m = build_lit(cfg_for(DATASETS[0]))
dev = m._generation_device()
states = []  # (контекст без якоря, якорь, 31 целевой токен)
for ds in DATASETS:
    for _, _, ids in dataset_prompts(m, cfg_for(ds)):
        ar = m.ar_generate(input_ids=ids, max_new_tokens=SPACING * (P - 1) + 1 + DRAFT,
                           eos_token_id=m.tokenizer.eos_token_id)
        full = torch.cat([ids.to(dev), torch.tensor([ar["new_tokens"]], device=dev)], 1)
        for j in range(P):
            t = ids.shape[1] + SPACING * j  # позиция якоря
            if t + DRAFT >= full.shape[1]:
                break
            states.append((full[:, :t], full[0, t].item(), full[0, t + 1: t + 1 + DRAFT]))
del m
print(f"состояний: {len(states)}", flush=True)

# 2. Модель авторов с eager-вниманием: подменяем функцию внимания ТОЛЬКО в
# диффузионной ветке (AR-ветка берёт свою функцию из модуля transformers).
from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

T = AutoModelForCausalLM.from_pretrained("chiennv/Orthrus-Qwen3-1.7B", dtype=torch.float32,
                                         trust_remote_code=True, attn_implementation="eager")
T = T.to(dev).eval()
mod = sys.modules[type(T.model.layers[0].self_attn).__module__]
n_layers = T.config.num_hidden_layers
groups = T.config.num_attention_heads // T.config.num_key_value_heads

MODE = {"kind": None, "layers": set()}
STATS = {L: {"spread": [], "maxdev": [], "sink": [], "small": [], "mass_out": [],
             "e2": [], "e1": [], "e0": []} for L in range(n_layers)}


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    k = key.repeat_interleave(groups, dim=1).float()
    v = value.repeat_interleave(groups, dim=1).float()
    s = (query.float() @ k.transpose(2, 3)) * scaling  # [B, H, T, n]
    if attention_mask is not None:
        raise RuntimeError("в диффузионном проходе маски быть не должно")
    p = s.softmax(-1)
    o = p @ v
    n = s.shape[-1]
    dev_s = s - s.mean(-1, keepdim=True)
    o1 = ((1.0 + dev_s) / n) @ v
    w2 = 1.0 + dev_s + 0.5 * dev_s ** 2
    o2 = (w2 / w2.sum(-1, keepdim=True)) @ v
    o0 = v.mean(-2, keepdim=True).expand_as(o)
    L = module.layer_idx
    if MODE["kind"] == "stats":
        st = STATS[L]
        norm = o.norm(dim=-1).clamp_min(1e-12)
        st["spread"].append(dev_s.std(-1).mean().item())
        st["maxdev"].append(dev_s.max(-1).values.mean().item())
        st["sink"].append(p[..., 0].mean().item())
        big = dev_s.abs() > 1.5
        st["small"].append((~big).float().mean().item())
        st["mass_out"].append((p * big).sum(-1).mean().item())
        st["e2"].append(((o2 - o).norm(dim=-1) / norm).mean().item())
        st["e1"].append(((o1 - o).norm(dim=-1) / norm).mean().item())
        st["e0"].append(((o0 - o).norm(dim=-1) / norm).mean().item())
    if L in MODE["layers"]:
        if MODE["kind"] == "taylor":
            o = o1 if ORDER == 1 else o2
        elif MODE["kind"] == "drop":
            o = torch.zeros_like(o)
    return o.to(query.dtype).transpose(1, 2).contiguous(), None


mod.eager_attention_forward = patched
mask_id = T.config.mask_token_id

caches = []
with torch.no_grad():
    for ctx, _, _ in states:
        c = DynamicCache(config=T.config)
        T(input_ids=ctx, position_ids=torch.arange(ctx.shape[1], device=dev)[None],
          past_key_values=c, use_cache=True)
        caches.append(c)


def acceptance(kind=None, layers=()):
    MODE["kind"], MODE["layers"] = kind, set(layers)
    accs = []
    with torch.no_grad():
        for (ctx, anchor, target), c in zip(states, caches):
            L = ctx.shape[1]
            blk = torch.full((1, DRAFT + 1), mask_id, device=dev)
            blk[0, 0] = anchor
            out = T(input_ids=blk, position_ids=torch.arange(L, L + DRAFT + 1, device=dev)[None],
                    past_key_values=c, use_cache=False, is_diffusion_pass=True, ar_seq_len=L)
            draft = out.logits[0, :-1].argmax(-1)
            accs.append(int((draft == target).long().cumprod(0).sum()))
    return sum(accs) / len(accs)


base = acceptance("stats")
print(json.dumps({"базовая приёмка": round(base, 3)}), flush=True)
rows = []
for L in range(n_layers):
    st = {k: sum(v) / len(v) for k, v in STATS[L].items()}
    st["acc_taylor"] = acceptance("taylor", [L])
    if ORDER == 1:
        st["acc_drop"] = acceptance("drop", [L])
    rows.append(st)
    print(json.dumps({"слой": L, **{k: round(x, 3) for k, x in st.items()}}), flush=True)

# 3. Гибрид: линеаризуем k слоёв, у которых замена рядом стоит меньше всего.
order = sorted(range(n_layers), key=lambda L: base - rows[L]["acc_taylor"])
for k in (2, 4, 8, 12, 16, 28):
    print(json.dumps({"порядок": ORDER, "линеаризовано слоёв": k, "какие": sorted(order[:k]),
                      "приёмка": round(acceptance("taylor", order[:k]), 3)}), flush=True)
