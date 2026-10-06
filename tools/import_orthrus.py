"""Переложить выпущенный Orthrus-Qwen3 (HF, chiennv/*) в чекпоинт нашей обвязки.

  PYTHONPATH=. python tools/import_orthrus.py chiennv/Orthrus-Qwen3-1.7B out.ckpt
  PYTHONPATH=. python tools/import_orthrus.py --dry Qwen/Qwen3-0.6B out.ckpt   # без их весов

Их диффузионный путь — шесть «двойников» на слой: q/k/v/o_proj_diff и
q/k_norm_diff. У нас те же двойники живут в ``orthrus.df_weights`` под
именами исходных модулей, поэтому перекладка — это переименование
``<модуль>.weight`` -> ``<модуль>_diff.weight``. Маска у них — строка
``mask_token_id`` замороженной таблицы эмбеддингов, у нас — отдельный
параметр ``mask_embedding``; кладём туда ту же строку.

Ствол берётся у Qwen по имени, как во всей обвязке, а не из их файла.
Поэтому сверяем: если их AR-веса отличаются от Qwen, «без потерь» у нас и у
них означало бы разное, и сравнение приёмки было бы нечестным.
"""
import os
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DRY = "--dry" in sys.argv
args = [a for a in sys.argv[1:] if a != "--dry"]
REPO, OUT = args[0], args[1]
DTYPE = os.environ.get("FD_DTYPE", "float32")
BASE = REPO if DRY else "Qwen/Qwen3-" + REPO.rsplit("-", 1)[-1]

overrides = [
    "+experiment=qwen_orthrus",
    f"model.name={BASE}",
    # Их набор двойников целиком: статья называет только Q, K, V, но в
    # выпущенных весах обучены ещё O и обе нормы.
    "model.adapter.w_names=[q_proj,k_proj,v_proj,o_proj,q_norm,k_norm]",
    f"model.backbone.dtype={DTYPE}",
    "model.backbone.device_map=null",
]
with initialize_config_dir(config_dir=os.path.join(ROOT, "src", "configs"),
                           version_base="1.3"):
    cfg = compose("train", overrides=overrides)

from src.models.orthrus import Orthrus  # noqa: E402

model = Orthrus(cfg)
adapter = model.orthrus
names = list(adapter._df_names)
print(f"двойников у нас: {len(names)}, например {names[:2]}", flush=True)

if not DRY:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json

    sd = load_file(hf_hub_download(REPO, "model.safetensors"))
    mask_id = json.load(open(hf_hub_download(REPO, "config.json")))["mask_token_id"]

    with torch.no_grad():
        for twin, name in zip(adapter.df_weights, names):
            module, param = name.rsplit(".", 1)
            src = sd[f"{module}_diff.{param}"]
            if src.shape != twin.shape:
                raise ValueError(f"{name}: у них {tuple(src.shape)}, у нас {tuple(twin.shape)}")
            twin.copy_(src.float())
        adapter.mask_embedding.copy_(sd["model.embed_tokens.weight"][mask_id].float()[None])

        # Сверка ствола: их AR-веса против Qwen, загруженного по имени.
        worst, n_cmp, missing = 0.0, 0, []
        for name, p in adapter.model.named_parameters():
            if name in sd:
                worst = max(worst, (p.float() - sd[name].float()).abs().max().item())
                n_cmp += 1
            elif name != "lm_head.weight":
                missing.append(name)
        print(f"ствол: сверено {n_cmp} тензоров, макс. расхождение {worst:.3e}, "
              f"нет в их файле: {missing[:3]}", flush=True)
        # Насколько их двойники ушли от инициализации копией AR — то есть что
        # они действительно обучали. Статья говорит «только Q, K, V».
        for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            moved = [
                ((sd[f"{n.rsplit('.', 1)[0]}_diff.weight"].float()
                  - sd[n].float()).norm() / sd[n].float().norm()).item()
                for n in sd if n.endswith(f"self_attn.{suffix}.weight")
            ]
            print(f"  {suffix}_diff: относительный сдвиг от AR {sum(moved) / len(moved):.4f}",
                  flush=True)
        # Полное обучение или LoRA: у LoRA ранга r разность весов лежит в r
        # направлениях. В их config.json есть поля r и lora_alpha.
        for layer in (0, len(adapter.model.model.layers) - 1):
            for m in ("q_proj", "o_proj"):
                base = sd[f"model.layers.{layer}.self_attn.{m}.weight"].float()
                diff = sd[f"model.layers.{layer}.self_attn.{m}_diff.weight"].float() - base
                energy = torch.linalg.svdvals(diff).pow(2).cumsum(0)
                energy = energy / energy[-1]
                print(f"  слой {layer} {m}: ранг на 90% энергии "
                      f"{int((energy < 0.9).sum()) + 1}/{min(base.shape)}, "
                      f"доля 16 главных {energy[15]:.2f}", flush=True)

trainable = {n for n, p in model.named_parameters() if p.requires_grad}
state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}
print(f"сохраняю {len(state)} обучаемых тензоров", flush=True)
torch.save(
    {"state_dict": state,
     "hyper_parameters": OmegaConf.to_container(cfg, resolve=False),
     "global_step": 0, "epoch": 0,
     "campaign_metadata": {"source": REPO}},
    OUT,
)
print(f"готово: {OUT}", flush=True)
