"""Где обученный двойник внимания можно заменить копией AR-весов.

  PYTHONPATH=. python tools/twin_copy_probe.py <ckpt> <наборы через запятую> <промптов> <якорей>

Состояния строятся так же, как в `tools/taylor_attention_probe.py`: промпт и
жадное продолжение нашей обвязкой, якорь через каждые 8 токенов, 31 целевой
токен. Приёмка состояния — длина совпадающего префикса argmax черновика с
жадным продолжением.

По каждому слою модели авторов (`chiennv/Orthrus-Qwen3-1.7B`):
  shift     — средний относительный сдвиг двойников q/k/v/o от AR-весов;
  acc_copy  — приёмка, если в ЭТОМ слое диффузионный вид берёт AR-веса
              (q, k, v, o и обе нормы) вместо обученных двойников.
Затем гибрид: копии в k слоях, где замена стоит меньше всего.
"""
import json
import os
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.getcwd())

CK, DATASETS, N, P = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4])
SPACING, DRAFT = 8, 31

from src.models.factory import build_lit  # noqa: E402
from src.eval import dataset_prompts  # noqa: E402

cfg_dir = os.path.abspath("src/configs")


def cfg_for(ds):
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        return compose("eval", overrides=[f"checkpoint={CK}", f"data={ds}",
                                          f"decode.n_prompts={N}", "model.backbone.dtype=float32"])


m = build_lit(cfg_for(DATASETS[0]))
dev = m._generation_device()
states = []
for ds in DATASETS:
    for _, _, ids in dataset_prompts(m, cfg_for(ds)):
        ar = m.ar_generate(input_ids=ids, max_new_tokens=SPACING * (P - 1) + 1 + DRAFT,
                           eos_token_id=m.tokenizer.eos_token_id)
        full = torch.cat([ids.to(dev), torch.tensor([ar["new_tokens"]], device=dev)], 1)
        for j in range(P):
            t = ids.shape[1] + SPACING * j
            if t + DRAFT >= full.shape[1]:
                break
            states.append((full[:, :t], full[0, t].item(), full[0, t + 1: t + 1 + DRAFT]))
del m
print(f"состояний: {len(states)}", flush=True)

from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

T = AutoModelForCausalLM.from_pretrained("chiennv/Orthrus-Qwen3-1.7B", dtype=torch.float32,
                                         trust_remote_code=True, attn_implementation="sdpa")
T = T.to(dev).eval()
mask_id = T.config.mask_token_id
n_layers = T.config.num_hidden_layers
PAIRS = [("q_proj", "q_proj_diff"), ("k_proj", "k_proj_diff"), ("v_proj", "v_proj_diff"),
         ("o_proj", "o_proj_diff"), ("q_norm", "q_norm_diff"), ("k_norm", "k_norm_diff")]

caches = []
with torch.no_grad():
    for ctx, _, _ in states:
        c = DynamicCache(config=T.config)
        T(input_ids=ctx, position_ids=torch.arange(ctx.shape[1], device=dev)[None],
          past_key_values=c, use_cache=True)
        caches.append(c)


def acceptance():
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


def with_copies(layers):
    saved = []
    for L in layers:
        attn = T.model.layers[L].self_attn
        for ar_name, diff_name in PAIRS:
            d = getattr(attn, diff_name).weight
            saved.append((d, d.data.clone()))
            d.data.copy_(getattr(attn, ar_name).weight.data)
    try:
        return acceptance()
    finally:
        for d, w in saved:
            d.data.copy_(w)


base = acceptance()
print(json.dumps({"базовая приёмка": round(base, 3)}), flush=True)
rows = []
for L in range(n_layers):
    attn = T.model.layers[L].self_attn
    shift = sum(((getattr(attn, d).weight - getattr(attn, a).weight).norm()
                 / getattr(attn, a).weight.norm()).item() for a, d in PAIRS[:4]) / 4
    acc = with_copies([L])
    rows.append(acc)
    print(json.dumps({"слой": L, "shift": round(shift, 3), "acc_copy": round(acc, 3)}), flush=True)

order = sorted(range(n_layers), key=lambda L: base - rows[L])
for k in (2, 4, 8, 12, 16, 28):
    print(json.dumps({"копий слоёв": k, "какие": sorted(order[:k]),
                      "приёмка": round(with_copies(order[:k]), 3)}), flush=True)
