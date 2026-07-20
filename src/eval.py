import json
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.models.factory import build_lit


def append_jsonl(path: Path, rows) -> None:
    """Append complete JSON lines under an advisory inter-process lock."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        try:
            for row in rows:
                output.write(json.dumps(row) + "\n")
            output.flush()
        finally:
            fcntl.flock(output.fileno(), fcntl.LOCK_UN)


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
                    temperature=0.0, top_k=None, top_p=None, coupled=True,
                    eos_token_id=None):
    """model.generate vs model.ar_generate on one prompt -> metrics.

    ``lossless`` is checked bitwise at ``temperature=0`` AND at
    ``temperature>0`` with Gumbel-coupled sampling (the default). Only the
    uncoupled (Leviathan) mode is equal in distribution rather than
    token-for-token — there ``lossless`` is None and the TV equivalence
    check applies instead.
    """
    sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p, coupled=coupled)
    fd = model.generate(
        input_ids=prompt_ids, block_size=block_size, jumps=jumps,
        max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, **sampling,
    )
    ar = model.ar_generate(
        input_ids=prompt_ids, max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id, **sampling,
    )
    n_tokens = len(fd["new_tokens"])
    bitwise_applicable = temperature == 0 or coupled
    lossless = fd["new_tokens"] == ar["new_tokens"] if bitwise_applicable else None
    divergence = None
    if lossless is False:
        common = min(len(fd["new_tokens"]), len(ar["new_tokens"]))
        divergence = next(
            (i for i in range(common) if fd["new_tokens"][i] != ar["new_tokens"][i]),
            common,
        )
    return {
        "lossless": lossless,
        # HEADLINE metrics — hardware/kernel independent:
        # drafted tokens accepted per cycle (what the whole project optimizes)
        "acceptance": (
            sum(fd["acceptance"]) / len(fd["acceptance"])
            if fd["acceptance"] else 0.0
        ),
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
        "_diagnostic": {
            "first_divergence": divergence,
            "flowdraft_token": (
                fd["new_tokens"][divergence]
                if divergence is not None and divergence < len(fd["new_tokens"])
                else None
            ),
            "ar_token": (
                ar["new_tokens"][divergence]
                if divergence is not None and divergence < len(ar["new_tokens"])
                else None
            ),
            "flowdraft_length": len(fd["new_tokens"]),
            "ar_length": len(ar["new_tokens"]),
        },
    }


def aggregate(per_prompt):
    """Mean ± std of every numeric metric (acceptance varies a lot across
    prompts — a mean without spread is not a reportable number); ``None``
    values are skipped per-key. ``lossless`` must hold on EVERY prompt
    (None = sampling mode, bitwise equality not applicable)."""
    out = {}
    for key in per_prompt[0]:
        if key == "lossless" or key.startswith("_"):
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


@torch.no_grad()
def sampling_equivalence(model, prompt_ids, *, n_samples, block_size, jumps,
                         temperature, top_k=None, top_p=None):
    """Distributional lossless check for ``temperature > 0``.

    Bitwise comparison is impossible without coupling the RNG streams of the
    speculative and AR paths, so equality of LAWS is tested instead:
    null-calibrated total variation on first-token counts. ``tv_null`` —
    the TV between two independent AR runs (pure sampling noise); the
    speculative path passes if ``tv_fd_ar`` does not exceed it materially.
    """
    from collections import Counter

    c_fd, c_ar1, c_ar2 = Counter(), Counter(), Counter()
    for _ in range(n_samples):
        # coupled=False is ESSENTIAL: this test targets the uncoupled
        # (Leviathan) mode — with the coupled default every call would be
        # deterministic (same seed), all three counters would collapse to a
        # single token and the test would pass vacuously (TV = 0 = 0).
        c_fd[model.generate(
            input_ids=prompt_ids, block_size=block_size, jumps=jumps, max_new_tokens=1,
            temperature=temperature, top_k=top_k, top_p=top_p, coupled=False,
        )["new_tokens"][0]] += 1
        for c in (c_ar1, c_ar2):
            c[model.ar_generate(
                input_ids=prompt_ids, max_new_tokens=1,
                temperature=temperature, top_k=top_k, top_p=top_p, coupled=False,
            )["new_tokens"][0]] += 1

    def tv(a, b):
        return sum(abs(a[k] - b[k]) for k in set(a) | set(b)) / (2 * n_samples)

    return {"tv_fd_ar": tv(c_fd, c_ar1), "tv_null": tv(c_ar1, c_ar2)}


def dataset_prompts(model, cfg):
    """The first ``decode.n_prompts`` validation samples of ``cfg.data`` —
    the full rendered prompt each (``decode.prompt_len=N`` switches to
    N-token prefixes). ``data=math500`` (default) is a benchmark the drafter
    never saw in training; ``data=nemotron`` reads the training
    distribution's val slice. Yields ``(source_index, label, prompt_ids [1, P])``.

    For ad-hoc prompts use ``main.py`` — evaluation is dataset-only.
    """
    from src.data import build_dataloaders

    train_loader, val_loader = build_dataloaders(cfg, model.tokenizer, model.df_processor)
    loader = val_loader if val_loader is not None else train_loader
    n_prompts = cfg.decode.get("n_prompts", 8)
    prompt_offset = int(cfg.decode.get("prompt_offset", 0))
    if prompt_offset < 0:
        raise ValueError("decode.prompt_offset must be non-negative")
    prompt_len = cfg.decode.get("prompt_len", None) or 10**9  # null -> full prompt
    seen = 0
    taken = 0
    for batch in loader:
        ids, mask = batch["input_ids"], batch["attention_mask"]
        for i in range(ids.size(0)):
            live = int(mask[i].sum())
            prompt = ids[i : i + 1, : min(live, prompt_len)].to(
                model._generation_device()
            )
            if prompt.size(1) < 2:  # no usable context
                continue
            source_index = seen
            seen += 1
            if source_index < prompt_offset:
                continue
            if model.tokenizer is not None:
                label = model.tokenizer.decode(prompt[0], skip_special_tokens=True)[:40]
            else:
                label = f"sample {taken}"
            yield source_index, label, prompt
            taken += 1
            if n_prompts is not None and taken >= n_prompts:
                return


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from src.data import quiet_download_logs

    quiet_download_logs()
    torch.manual_seed(cfg.seed)
    model = build_lit(cfg)

    dec = cfg.decode
    results = []
    prompt_rows = []
    first_prompt = None
    for source_index, label, ids in dataset_prompts(model, cfg):
        if first_prompt is None:
            first_prompt = ids
        context_window = getattr(model.orthrus.model.config, "max_position_embeddings", None)
        if context_window is not None and ids.size(1) + dec.max_new_tokens > context_window:
            raise ValueError(
                f"prompt index {source_index} has {ids.size(1)} tokens and requests "
                f"{dec.max_new_tokens} new tokens, exceeding the model context "
                f"window {context_window}; set decode.prompt_len explicitly"
            )
        metrics = evaluate_prompt(
            model, ids,
            block_size=dec.block_size,
            jumps=dec.jumps,
            max_new_tokens=dec.max_new_tokens,
            temperature=dec.get("temperature", 0.0),
            top_k=dec.get("top_k", None),
            top_p=dec.get("top_p", None),
            coupled=dec.get("coupled", True),
            eos_token_id=model.tokenizer.eos_token_id,
        )
        logger.info(f"{label!r}: {metrics}")
        results.append(metrics)
        prompt_rows.append(
            {
                "prompt_index": source_index,
                "prompt_label": label,
                **{key: value for key, value in metrics.items() if not key.startswith("_")},
                **metrics["_diagnostic"],
            }
        )

    if not results:
        raise RuntimeError(f"benchmark {cfg.data.dataset!r} produced no usable prompts")
    summary = aggregate(results)
    logger.info(
        f"=== block_size={dec.block_size} jumps={dec.jumps} "
        f"temperature={dec.get('temperature', 0.0)} ==="
    )
    logger.info(OmegaConf.to_yaml(summary))

    # Machine-readable aggregate + per-prompt rows. The shared metadata makes
    # every record independently attributable after concurrent campaign runs.
    from hydra.utils import to_absolute_path

    train_cfg = cfg.get("train", {})
    row = {
        "run_id": cfg.get("run_id", None),
        "experiment_id": cfg.get("experiment_id", None),
        "split_label": cfg.get("split_label", None),
        "eval_seed": cfg.seed,
        "training_seed": getattr(model, "checkpoint_seed", None),
        "training_run_name": getattr(model, "checkpoint_run_name", None),
        "variant": cfg.get("variant", "flowdraft"),
        "model": cfg.model.name,
        "dataset": cfg.data.get("benchmark", cfg.data.dataset),
        "checkpoint": cfg.checkpoint,
        "checkpoint_step": getattr(model, "checkpoint_global_step", None),
        "training_elapsed_seconds": getattr(model, "checkpoint_elapsed_seconds", None),
        "training_device_hours": getattr(model, "checkpoint_device_hours", None),
        "attention_backend": cfg.model.backbone.get("attn_implementation", None),
        "train_lr": train_cfg.get("lr", None),
        "train_time_sampling": train_cfg.get("time_sampling", None),
        "train_lambda": train_cfg.get("lambda", None),
        "train_ar_kl_weight": train_cfg.get("ar_kl_weight", None),
        "train_anchor_point": train_cfg.get("anchor_point", None),
        "block_size": dec.block_size,
        "jumps": dec.jumps if isinstance(dec.jumps, int) else list(dec.jumps),
        "temperature": dec.get("temperature", 0.0),
        "coupled": dec.get("coupled", True),
        "n_prompts": len(results),
        "prompt_offset": int(dec.get("prompt_offset", 0)),
        "max_new_tokens": dec.max_new_tokens,
        **summary,
    }
    results_file = cfg.get("results_file", None)
    if results_file:
        path = Path(to_absolute_path(str(results_file)))
        append_jsonl(path, [row])
        logger.info(f"row appended -> {path}")
    per_prompt_file = cfg.get("per_prompt_file", None)
    if per_prompt_file:
        prompt_path = Path(to_absolute_path(str(per_prompt_file)))
        append_jsonl(
            prompt_path, ({**row, **prompt_row} for prompt_row in prompt_rows)
        )
        logger.info(f"{len(prompt_rows)} prompt rows appended -> {prompt_path}")
    lossless_policy = cfg.get("lossless_policy", "assert")
    if lossless_policy not in {"assert", "diagnose", "ignore"}:
        raise ValueError("lossless_policy must be assert, diagnose, or ignore")
    if summary["lossless"] is False and lossless_policy == "assert":
        raise RuntimeError("LOSSLESS CHECK FAILED — flow-draft output diverged from greedy AR")
    if summary["lossless"] is False and lossless_policy == "diagnose":
        logger.warning("lossless divergence recorded (diagnostic policy; run not eligible for selection)")

    # UNCOUPLED sampling: bitwise equality is N/A, equality of LAWS is asserted
    temperature = dec.get("temperature", 0.0)
    equiv_samples = dec.get("equiv_samples", 0)
    if temperature > 0 and not dec.get("coupled", True) and equiv_samples > 0 and first_prompt is not None:
        eq = sampling_equivalence(
            model, first_prompt, n_samples=equiv_samples,
            block_size=dec.block_size, jumps=dec.jumps,
            temperature=temperature, top_k=dec.get("top_k"), top_p=dec.get("top_p"),
        )
        logger.info(f"sampling equivalence: TV(fd,ar)={eq['tv_fd_ar']:.3f} vs noise TV(ar,ar')={eq['tv_null']:.3f}")
        if eq["tv_fd_ar"] > eq["tv_null"] + 0.05:
            raise RuntimeError("SAMPLING LOSSLESS CHECK FAILED — speculative law diverged from AR")


if __name__ == "__main__":
    main()
