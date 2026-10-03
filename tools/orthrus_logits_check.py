"""Один диффузионный проход двумя реализациями на одном состоянии: насколько
расходятся логиты черновика и где расходится их argmax.

  PYTHONPATH=. python tools/orthrus_logits_check.py <ckpt> <набор> <N промптов> <точек на промпт>

Состояние = промпт + жадное AR-продолжение длиной L; черновик блока K=32 от
якоря в позиции L. У нас — `_draft_block`, у авторов — forward с
is_diffusion_pass=True над их кэшем.
"""
import os, sys, json, torch
sys.path.insert(0, os.getcwd())
from hydra import compose, initialize_config_dir
CK, DATA, N, P = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with initialize_config_dir(config_dir=os.path.abspath("src/configs"), version_base="1.3"):
    cfg = compose("eval", overrides=[f"checkpoint={CK}", f"data={DATA}", f"decode.n_prompts={N}",
                  "model.backbone.dtype=float32"])
from src.models.factory import build_lit
from src.eval import dataset_prompts
from transformers import DynamicCache, AutoModelForCausalLM
m = build_lit(cfg); dev = m._generation_device()
states = []
for _, _, ids in dataset_prompts(m, cfg):
    ar = m.ar_generate(input_ids=ids, max_new_tokens=8 * P + 2, eos_token_id=m.tokenizer.eos_token_id)
    full = torch.cat([ids.to(dev), torch.tensor([ar["new_tokens"]], device=dev)], 1)
    for j in range(P):
        states.append(full[:, : ids.shape[1] + 8 * j + 1])
ours = []
with torch.no_grad():
    for s in states:
        cache = DynamicCache(config=m.orthrus.model.config)
        m.orthrus(s[:, :-1], torch.ones_like(s[:, :-1]), past_key_values=cache)
        _, q = m._draft_block(cache, 32, [(0.0, 1.0)], anchor_token=s[:, -1])
        ours.append(q[0].log().cpu())
del m
T = AutoModelForCausalLM.from_pretrained("chiennv/Orthrus-Qwen3-1.7B", dtype=torch.float32,
        trust_remote_code=True, attn_implementation="sdpa").to(dev).eval()
mask_id = T.config.mask_token_id
worst, flips, total, flip_margins = 0.0, 0, 0, []
with torch.no_grad():
    for s, lo in zip(states, ours):
        cache = DynamicCache(config=T.config); L = s.shape[1] - 1
        T(input_ids=s[:, :-1], position_ids=torch.arange(L, device=dev)[None], past_key_values=cache, use_cache=True)
        blk = torch.full((1, 32), mask_id, device=dev); blk[0, 0] = s[0, -1]
        out = T(input_ids=blk, position_ids=torch.arange(L, L + 32, device=dev)[None],
                past_key_values=cache, use_cache=False, is_diffusion_pass=True, ar_seq_len=L)
        lt = out.logits[0, :-1].float().log_softmax(-1).cpu()
        worst = max(worst, (lt - lo).abs().max().item())
        a_o, a_t = lo.argmax(-1), lt.argmax(-1)
        for i in torch.nonzero(a_o != a_t).flatten().tolist():
            top2 = lt[i].topk(2).values
            flip_margins.append(round((top2[0] - top2[1]).item(), 5))
        flips += int((a_o != a_t).sum()); total += a_o.numel()
print(json.dumps({"набор": DATA, "состояний": len(states), "позиций": total,
                  "макс |Δ log p|": round(worst, 6), "argmax разошёлся": flips,
                  "зазор top1-top2 у них в местах расхождения": flip_margins}))
