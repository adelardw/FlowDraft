"""Поцикловая сверка обвязки: generate авторов Orthrus против нашего.

  PYTHONPATH=. python tools/orthrus_xcheck.py <N> <max_new> <ckpt> <набор>

<ckpt> — их веса, переложенные `tools/import_orthrus.py`. Одни и те же N
промптов набора декодируются жадно двумя путями: нашим `generate` и
`generate` из их `modeling_orthrus.py` (chiennv/Orthrus-Qwen3-1.7B). У них
приёмка цикла снимается обёрткой forward: argmax диффузионного прохода
против argmax проверочного, длина совпадающего префикса — ровно их формула.
Печатает по строке на промпт: совпал ли текст и обе поцикловые последовательности.
"""
import os, sys, json, torch
sys.path.insert(0, os.getcwd())
from hydra import compose, initialize_config_dir
N, MAXNEW, CK, DATA = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
with initialize_config_dir(config_dir=os.path.abspath("src/configs"), version_base="1.3"):
    cfg = compose("eval", overrides=[f"checkpoint={CK}", f"data={DATA}", f"decode.n_prompts={N}",
        "decode.block_size=32", f"decode.max_new_tokens={MAXNEW}", "decode.jumps=1",
        "model.backbone.dtype=float32"])
from src.models.factory import build_lit
from src.eval import dataset_prompts
m = build_lit(cfg)
prompts, ours = [], []
for idx, label, ids in dataset_prompts(m, cfg):
    fd = m.generate(input_ids=ids, block_size=32, jumps=1, max_new_tokens=MAXNEW,
                    eos_token_id=m.tokenizer.eos_token_id, temperature=0.0)
    prompts.append(ids.cpu()); ours.append((fd["acceptance"], fd["new_tokens"]))
dev = m._generation_device(); del m
from transformers import AutoModelForCausalLM
T = AutoModelForCausalLM.from_pretrained("chiennv/Orthrus-Qwen3-1.7B", dtype=torch.float32,
    trust_remote_code=True, attn_implementation="sdpa").to(dev).eval()
st = {"draft": None, "acc": []}
fwd = T.forward
def hooked(*a, **k):
    out = fwd(*a, **k)
    if k.get("is_diffusion_pass"):
        st["draft"] = out.logits[:, :-1].argmax(-1)
    elif st["draft"] is not None and k.get("input_ids") is not None and k["input_ids"].shape[1] > 1:
        ar = out.logits.argmax(-1)[:, :-1]
        d = st["draft"][:, : ar.shape[1]]
        st["acc"].append(int((d == ar).long().cumprod(1).sum())); st["draft"] = None
    return out
T.forward = hooked
for ids, (acc_o, toks_o) in zip(prompts, ours):
    st["acc"], st["draft"] = [], None
    out = T.generate(input_ids=ids.to(dev), max_new_tokens=MAXNEW, temperature=0.0)
    toks_t = out[0, ids.shape[1]:].tolist()
    same = toks_t[:len(toks_o)] == toks_o[:len(toks_t)]
    print(json.dumps({"same_text": same, "n_ours": len(toks_o), "n_theirs": len(toks_t),
        "ours": acc_o, "theirs": st["acc"]}), flush=True)
