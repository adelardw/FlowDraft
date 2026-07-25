"""Benchmark the OFFICIAL Orthrus implementation, writing our result schema.

The other half of the validation 2x2. ``src/eval.py`` measures our decode loop;
this measures ``chiennv2000/orthrus``'s ``OrthrusLM.generate`` on the same
prompts, with the same metric definitions, appending to the same JSONL — so one
table can hold both codebases.

Their ``generate`` is never modified or copied. It returns only token ids, so
everything is recovered from a trace of ``forward`` calls:

* their diffusion pass is the only caller that passes ``ar_seq_len``, and it
  passes exactly the loop's ``start_idx`` (their ``model.py:473``), so the
  sequence of ``ar_seq_len`` values gives the exact number of tokens each
  cycle committed — its first difference minus one is that cycle's accepted
  draft length, the same quantity ``eval.py`` calls ``acceptance``;
* every cycle costs one diffusion forward plus one verification forward, so
  the call count is the honest forward budget for TPF.

The final cycle is excluded from the acceptance mean because it is truncated by
``max_new_tokens`` or by EOS and would bias the average downward.

    uv run python src/eval_reference.py \
        reference_path=weights/Orthrus-Qwen3-1.7B data=humaneval decode.block_size=32

IMPORTANT — ``tokens_per_s`` and ``speedup`` are comparable only WITHIN a
codebase. The reference AR baseline is HF ``GenerationMixin.generate`` while
ours is a hand-rolled loop; comparing wall clock across the two measures Python
overhead, not the method. ``acceptance``, ``tpf`` and ``lossless`` are
implementation-independent and are the cross-codebase numbers.
"""

import time
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.eval import aggregate, append_jsonl


class PromptSource:
    """Minimal stand-in for a LightningModule so ``eval.dataset_prompts`` can be
    reused verbatim — identical prompts across codebases is the precondition for
    the whole comparison, so the slicing must not be re-implemented here."""

    def __init__(self, tokenizer, df_processor, device):
        self.tokenizer = tokenizer
        self.df_processor = df_processor
        self._device = device

    def _generation_device(self):
        return self._device


class ForwardTrace:
    """Record every ``forward`` call of the reference model for the duration of
    one generate(). Shadows the bound method on the instance; ``nn.Module.__call__``
    resolves ``self.forward`` through the instance dict, so their code is untouched."""

    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.diffusion_starts = []
        self._original = model.forward

    def __enter__(self):
        def traced(*args, **kwargs):
            self.calls += 1
            if kwargs.get("is_diffusion_pass") and kwargs.get("ar_seq_len") is not None:
                self.diffusion_starts.append(int(kwargs["ar_seq_len"]))
            return self._original(*args, **kwargs)

        self.model.forward = traced
        return self

    def __exit__(self, *exc):
        self.model.forward = self._original
        return False

    def acceptance(self):
        """Accepted draft tokens per completed cycle."""
        starts = self.diffusion_starts
        if len(starts) < 2:
            return []
        return [starts[i + 1] - starts[i] - 1 for i in range(len(starts) - 1)]


def _sampling_kwargs(temperature, top_k, top_p):
    """Their generate signature defaults to top_k=20/top_p=0.8; at temperature 0
    it takes the argmax branch and neither is read."""
    if temperature <= 0:
        return {"temperature": 0.0}
    return {
        "temperature": float(temperature),
        "top_k": int(top_k) if top_k else 0,
        "top_p": float(top_p) if top_p else 1.0,
    }


@torch.no_grad()
def evaluate_prompt(model, prompt_ids, *, max_new_tokens, temperature=0.0,
                    top_k=None, top_p=None, eos_token_id=None):
    """Their diffusion decode vs their own AR decode on one prompt."""
    sampling = _sampling_kwargs(temperature, top_k, top_p)
    prompt_len = prompt_ids.shape[1]

    with ForwardTrace(model) as diffusion:
        start = time.perf_counter()
        out = model.generate(
            input_ids=prompt_ids, max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id, use_diffusion_mode=True, **sampling,
        )
        seconds = time.perf_counter() - start
    tokens = out[0, prompt_len:].tolist()

    with ForwardTrace(model) as autoregressive:
        start = time.perf_counter()
        ar_out = model.generate(
            input_ids=prompt_ids, max_new_tokens=max_new_tokens,
            # Their AR branch delegates to HF generate, which infers a mask from
            # the pad token when none is given — and pad == eos here, so an
            # unset mask would make HF treat a generated eos as padding.
            attention_mask=torch.ones_like(prompt_ids),
            eos_token_id=eos_token_id, use_diffusion_mode=False,
            do_sample=temperature > 0, **sampling,
        )
        ar_seconds = time.perf_counter() - start
    ar_tokens = ar_out[0, prompt_len:].tolist()

    accepted = diffusion.acceptance()
    lossless = tokens == ar_tokens if temperature == 0 else None
    divergence = None
    if lossless is False:
        common = min(len(tokens), len(ar_tokens))
        divergence = next(
            (i for i in range(common) if tokens[i] != ar_tokens[i]), common
        )
    return {
        "lossless": lossless,
        "acceptance": sum(accepted) / len(accepted) if accepted else 0.0,
        "tpf": len(tokens) / diffusion.calls,
        "tpf_ar": len(ar_tokens) / autoregressive.calls,
        "tokens_per_s": len(tokens) / seconds,
        "tokens_per_s_ar": len(ar_tokens) / ar_seconds,
        "speedup": (len(tokens) / seconds) / (len(ar_tokens) / ar_seconds),
        "nll": None,
        "n_tokens": len(tokens),
        "_diagnostic": {
            "first_divergence": divergence,
            "flowdraft_token": (
                tokens[divergence]
                if divergence is not None and divergence < len(tokens) else None
            ),
            "ar_token": (
                ar_tokens[divergence]
                if divergence is not None and divergence < len(ar_tokens) else None
            ),
            "flowdraft_length": len(tokens),
            "ar_length": len(ar_tokens),
            "cycles": len(diffusion.diffusion_starts),
        },
    }


def build_reference(cfg: DictConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.preprocessor.df_processor import DiffusionProcessor

    path = cfg.get("reference_path")
    if not path:
        raise ValueError(
            "reference_path is required: point it at an Orthrus release "
            "directory, e.g. reference_path=weights/Orthrus-Qwen3-1.7B"
        )
    from hydra.utils import to_absolute_path

    resolved = Path(to_absolute_path(str(path)))
    if not resolved.is_dir():
        raise FileNotFoundError(f"reference release not found: {resolved}")

    backbone = cfg.model.backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        dtype=getattr(torch, str(backbone.get("dtype", "bfloat16"))),
        attn_implementation=str(backbone.get("attn_implementation", "sdpa")),
        trust_remote_code=True,
    ).to(device).eval()

    # Their block includes the clean anchor and is read from the config at
    # generate() time, which is what makes a matched-K comparison possible.
    model.config.block_size = int(cfg.decode.block_size)
    tokenizer = AutoTokenizer.from_pretrained(str(resolved))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = DiffusionProcessor.from_model(tokenizer, model)
    logger.info(
        f"reference OrthrusLM from {resolved} on {device} "
        f"({backbone.get('dtype')}, {backbone.get('attn_implementation')}, "
        f"block_size={model.config.block_size})"
    )
    return model, PromptSource(tokenizer, processor, device)


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from hydra.utils import to_absolute_path

    from src.data import quiet_download_logs
    from src.eval import dataset_prompts

    quiet_download_logs()
    torch.manual_seed(cfg.seed)
    model, source = build_reference(cfg)
    mask_token_id = getattr(model.config, "mask_token_id", None)

    dec = cfg.decode
    if int(dec.get("jumps", 1)) != 1:
        raise ValueError(
            "the reference drafter reconstructs a block in exactly one step; "
            "decode.jumps must be 1"
        )
    results, prompt_rows = [], []
    for source_index, label, ids in dataset_prompts(source, cfg):
        context_window = getattr(model.config, "max_position_embeddings", None)
        if context_window is not None and ids.size(1) + dec.max_new_tokens > context_window:
            raise ValueError(
                f"prompt index {source_index} has {ids.size(1)} tokens and requests "
                f"{dec.max_new_tokens} new tokens, exceeding the model context "
                f"window {context_window}; set decode.prompt_len explicitly"
            )
        metrics = evaluate_prompt(
            model, ids,
            max_new_tokens=dec.max_new_tokens,
            temperature=dec.get("temperature", 0.0),
            top_k=dec.get("top_k", None),
            top_p=dec.get("top_p", None),
            eos_token_id=source.tokenizer.eos_token_id,
        )
        # to-theirs writes our learned mask vector into the tied embedding row,
        # which in principle makes that id emittable. Assert it never is.
        if mask_token_id is not None and metrics["_diagnostic"]["flowdraft_length"]:
            emitted_mask = metrics["_diagnostic"].get("flowdraft_token") == mask_token_id
            if emitted_mask:
                raise RuntimeError(
                    f"mask token {mask_token_id} was emitted as output — the "
                    "converted embedding row is contaminating generation"
                )
        logger.info(f"{label!r}: {metrics}")
        results.append(metrics)
        prompt_rows.append({
            "prompt_index": source_index,
            "prompt_label": label,
            **{k: v for k, v in metrics.items() if not k.startswith("_")},
            **metrics["_diagnostic"],
        })

    if not results:
        raise RuntimeError(f"benchmark {cfg.data.dataset!r} produced no usable prompts")
    summary = aggregate(results)
    logger.info(f"=== reference block_size={dec.block_size} ===")
    logger.info(OmegaConf.to_yaml(summary))

    row = {
        "run_id": cfg.get("run_id", None),
        "experiment_id": cfg.get("experiment_id", None),
        "split_label": cfg.get("split_label", None),
        "codebase": "reference",
        "weights_source": cfg.get("weights_source", None),
        "eval_seed": cfg.seed,
        "training_seed": None,
        "training_run_name": None,
        "variant": "orthrus",
        "model": cfg.model.name,
        "dataset": cfg.data.get("benchmark", cfg.data.dataset),
        "checkpoint": str(cfg.get("reference_path")),
        "checkpoint_step": None,
        "training_elapsed_seconds": None,
        "training_device_hours": None,
        "attention_backend": cfg.model.backbone.get("attn_implementation", None),
        "block_size": dec.block_size,
        "jumps": 1,
        "temperature": dec.get("temperature", 0.0),
        "coupled": False,
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
        append_jsonl(prompt_path, ({**row, **r} for r in prompt_rows))
        logger.info(f"{len(prompt_rows)} prompt rows appended -> {prompt_path}")

    policy = cfg.get("lossless_policy", "assert")
    if summary["lossless"] is False and policy == "assert":
        raise RuntimeError(
            "LOSSLESS CHECK FAILED — reference diffusion output diverged from its own AR"
        )
    if summary["lossless"] is False and policy == "diagnose":
        logger.warning("lossless divergence recorded (diagnostic policy)")


if __name__ == "__main__":
    main()
