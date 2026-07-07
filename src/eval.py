import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.models.factory import build_lit


@torch.no_grad()
def continuation_nll(model, prompt_ids, new_tokens):
    """Generation quality: mean NLL of the continuation under the frozen AR
    teacher (identical for both decoders when greedy — the lossless property
    made measurable; also comparable across sampling temperatures)."""
    if not new_tokens:
        return float("nan")
    cont = torch.tensor([new_tokens], device=prompt_ids.device)
    full = torch.cat([prompt_ids, cont], dim=1)
    logits = model.orthrus(full, torch.ones_like(full)).logits
    log_p = logits[:, prompt_ids.size(1) - 1 : -1].float().log_softmax(-1)
    return float(-log_p.gather(-1, cont[..., None]).mean())


@torch.no_grad()
def evaluate_prompt(model, prompt_ids, *, block_size, jumps, max_new_tokens,
                    temperature=0.0, top_k=None, top_p=None, eos_token_id=None):
    """model.generate vs model.ar_generate on one prompt -> metrics.

    At ``temperature=0`` ``lossless`` is checked bitwise, not assumed. With
    sampling the outputs are equal in distribution, not token-for-token, so
    ``lossless`` is reported as None there.
    """
    sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p)
    fd = model.generate(
        input_ids=prompt_ids, block_size=block_size, jumps=jumps,
        max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, **sampling,
    )
    ar = model.ar_generate(
        input_ids=prompt_ids, max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id, **sampling,
    )
    n_tokens = len(fd["new_tokens"])
    assert fd["acceptance"], "generation ran zero cycles — check max_new_tokens"
    return {
        "lossless": fd["new_tokens"] == ar["new_tokens"] if temperature == 0 else None,
        # HEADLINE metrics — hardware/kernel independent:
        # drafted tokens accepted per cycle (what the whole project optimizes)
        "acceptance": sum(fd["acceptance"]) / len(fd["acceptance"]),
        # tokens per forward pass (cycle = jumps + 1 forwards; AR is ~1)
        "tpf": n_tokens / fd["n_forwards"],
        "tpf_ar": len(ar["new_tokens"]) / ar["n_forwards"],
        # wall-clock DIAGNOSTICS, not headline: hardware/kernel dependent
        # (default kernel is sdpa; try model.backbone.attn_implementation=flex_attention on GPU)
        "tokens_per_s": n_tokens / fd["seconds"],
        "tokens_per_s_ar": len(ar["new_tokens"]) / ar["seconds"],
        "speedup": (n_tokens / fd["seconds"]) / (len(ar["new_tokens"]) / ar["seconds"]),
        # teacher NLL of the continuation: meaningful under sampling only —
        # at temperature=0 the output is bitwise equal to AR (asserted above),
        # so the greedy NLL would measure nothing but the lossless property
        "nll": continuation_nll(model, prompt_ids, fd["new_tokens"]) if temperature > 0 else None,
        "n_tokens": n_tokens,
    }


def aggregate(per_prompt):
    """Mean ± std of every numeric metric (acceptance varies a lot across
    prompts — a mean without spread is not a reportable number); ``None``
    values are skipped per-key. ``lossless`` must hold on EVERY prompt
    (None = sampling mode, bitwise equality not applicable)."""
    out = {}
    for key in per_prompt[0]:
        if key == "lossless":
            continue
        vals = [r[key] for r in per_prompt if r[key] is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        out[key] = mean
        if len(vals) > 1:
            out[f"{key}_std"] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    flags = [r["lossless"] for r in per_prompt]
    out["lossless"] = None if all(f is None for f in flags) else all(f for f in flags if f is not None)
    return out


def dataset_prompts(model, cfg):
    """Prefixes of the first ``decode.n_prompts`` validation samples
    (``decode.prompt_len`` tokens each) from ``src.data.build_dataloaders`` —
    the same seam training reads, so acceptance is measured on the training
    distribution. Yields ``(label, prompt_ids [1, P])``.

    For ad-hoc prompts use ``main.py`` — evaluation is dataset-only.
    """
    from src.data import build_dataloaders

    train_loader, val_loader = build_dataloaders(cfg, model.tokenizer, model.df_processor)
    loader = val_loader if val_loader is not None else train_loader
    n_prompts = cfg.decode.get("n_prompts", 8)
    prompt_len = cfg.decode.get("prompt_len", 32)
    taken = 0
    for batch in loader:
        ids, mask = batch["input_ids"], batch["attention_mask"]
        for i in range(ids.size(0)):
            live = int(mask[i].sum())
            prompt = ids[i : i + 1, : min(live, prompt_len)].to(model.device)
            if prompt.size(1) < 2:  # no usable context
                continue
            if model.tokenizer is not None:
                label = model.tokenizer.decode(prompt[0], skip_special_tokens=True)[:40]
            else:
                label = f"sample {taken}"
            yield label, prompt
            taken += 1
            if taken >= n_prompts:
                return


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    model = build_lit(cfg)

    dec = cfg.decode
    results = []
    for label, ids in dataset_prompts(model, cfg):
        metrics = evaluate_prompt(
            model, ids,
            block_size=dec.block_size,
            jumps=dec.jumps,
            max_new_tokens=dec.max_new_tokens,
            temperature=dec.get("temperature", 0.0),
            top_k=dec.get("top_k", None),
            top_p=dec.get("top_p", None),
        )
        logger.info(f"{label!r}: {metrics}")
        results.append(metrics)

    summary = aggregate(results)
    logger.info(
        f"=== block_size={dec.block_size} jumps={dec.jumps} "
        f"temperature={dec.get('temperature', 0.0)} ==="
    )
    logger.info(OmegaConf.to_yaml(summary))
    if summary["lossless"] is False:
        raise RuntimeError("LOSSLESS CHECK FAILED — flow-draft output diverged from greedy AR")


if __name__ == "__main__":
    main()
