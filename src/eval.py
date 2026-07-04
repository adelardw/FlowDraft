import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf


@torch.no_grad()
def evaluate_prompt(model, prompt_ids, *, block_size, jumps, max_new_tokens, eos_token_id=None):
    """model.generate vs model.ar_generate on one prompt -> metrics.

    ``lossless`` is checked, not assumed: token-for-token equality of the
    flow-draft output with plain greedy AR.
    """
    fd = model.generate(
        input_ids=prompt_ids, block_size=block_size, jumps=jumps,
        max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
    )
    ar = model.ar_generate(
        input_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
    )
    n_tokens = len(fd["new_tokens"])
    return {
        "lossless": fd["new_tokens"] == ar["new_tokens"],
        # drafted tokens accepted per cycle — the number the whole project optimizes
        "acceptance": sum(fd["acceptance"]) / max(len(fd["acceptance"]), 1),
        # tokens per forward pass (AR baseline is ~1 by construction)
        "tpf": n_tokens / fd["n_forwards"],
        "tokens_per_s": n_tokens / fd["seconds"],
        "tokens_per_s_ar": len(ar["new_tokens"]) / ar["seconds"],
        "speedup": (n_tokens / fd["seconds"]) / (len(ar["new_tokens"]) / ar["seconds"]),
        "n_tokens": n_tokens,
    }


def aggregate(per_prompt):
    """Mean of every numeric metric; lossless must hold on EVERY prompt."""
    keys = [k for k in per_prompt[0] if k != "lossless"]
    out = {k: sum(r[k] for r in per_prompt) / len(per_prompt) for k in keys}
    out["lossless"] = all(r["lossless"] for r in per_prompt)
    return out


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from src.models.lit_orthrus import FlowMapOrthrus

    torch.manual_seed(cfg.seed)
    model = FlowMapOrthrus(cfg)
    if cfg.checkpoint:
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=False)  # ckpt holds the DF head only
        logger.info(f"DF head loaded from {cfg.checkpoint}")

    results = []
    for prompt in cfg.decode.prompts:
        ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        metrics = evaluate_prompt(
            model, ids,
            block_size=cfg.decode.block_size,
            jumps=cfg.decode.jumps,
            max_new_tokens=cfg.decode.max_new_tokens,
        )
        logger.info(f"{prompt[:40]!r}: {metrics}")
        results.append(metrics)

    summary = aggregate(results)
    logger.info(f"=== block_size={cfg.decode.block_size} jumps={cfg.decode.jumps} ===")
    logger.info(OmegaConf.to_yaml(summary))
    if not summary["lossless"]:
        raise RuntimeError("LOSSLESS CHECK FAILED — flow-draft output diverged from greedy AR")


if __name__ == "__main__":
    main()
