"""Do our decode geometry and the official Orthrus decode geometry agree?

Loads the SAME (authors') diffusion weights into both codebases, gives both an
identical AR prefix cache and an identical ``[anchor, MASK x (K-1)]`` block, runs
one diffusion forward each, and compares the resulting block proposal.

This is the whole 2x2 in miniature and it costs seconds instead of GPU-hours. If
it passes, a later acceptance gap between cell A and cell B is attributable to
training rather than to our implementation. If it fails, the grid would only be
measuring our bug, so the runner treats a failure here as fatal.

Runs on CPU in float32 on purpose: bf16 and fused attention kernels differ
between the two paths for reasons that have nothing to do with the geometry
under test, and would put noise where the signal should be.

    uv run python -m src.tools.equivalence_gate --block-size 32
"""

import argparse
import gc
from pathlib import Path

import torch
from loguru import logger

PROMPT = "Write a program to count the frequency of each word in a paragraph."
REFERENCE_DIR = "weights/Orthrus-Qwen3-1.7B"
CONVERTED_CKPT = "weights/orthrus-authors.ckpt"
# Their block includes the clean anchor, so K drafts K-1 fresh tokens
# (orthrus.py:280, model.py:461-463). Both sides use the same convention.
DEFAULT_BLOCK = 32


def _prompt_ids(reference: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(reference))
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": PROMPT}]
    ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
    ).input_ids
    return ids, tokenizer


@torch.no_grad()
def reference_block(reference: Path, prompt_ids, block_size: int):
    """One diffusion forward through the authors' unmodified OrthrusLM."""
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache

    model = AutoModelForCausalLM.from_pretrained(
        str(reference),
        dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()
    mask_token_id = model.config.mask_token_id
    prompt_len = prompt_ids.shape[1]
    cache = DynamicCache(config=model.config)

    prefill = model(
        input_ids=prompt_ids,
        position_ids=torch.arange(prompt_len).unsqueeze(0),
        past_key_values=cache,
    )
    anchor = prefill.logits[:, -1, :].argmax(-1, keepdim=True)

    block_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long)
    block_ids[:, 0] = anchor
    diffusion = model(
        input_ids=block_ids,
        position_ids=torch.arange(prompt_len, prompt_len + block_size).unsqueeze(0),
        past_key_values=cache,
        use_cache=False,
        is_diffusion_pass=True,
        ar_seq_len=prompt_len,
    )
    probs = diffusion.logits[:, :-1, :].float().softmax(-1)
    del model
    gc.collect()
    return anchor, probs


@torch.no_grad()
def our_block(checkpoint: str, prompt_ids, anchor, block_size: int):
    """The same forward through Orthrus._draft_block in our codebase."""
    import hydra
    from transformers.cache_utils import DynamicCache

    from src.models.factory import build_lit

    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(
            config_name="eval",
            overrides=[
                "model=qwen3_1.7b",
                f"checkpoint={checkpoint}",
                "variant=orthrus",
                "model.backbone.device_map=null",
                "model.backbone.dtype=float32",
                "model.backbone.attn_implementation=eager",
            ],
        )
    model = build_lit(cfg).to("cpu").eval()
    cache = DynamicCache(config=model.orthrus.model.config)
    model.orthrus(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        past_key_values=cache,
    )
    _, probs = model._draft_block(
        cache, block_size=block_size, times=[0.0, 1.0], anchor_token=anchor
    )
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=Path(REFERENCE_DIR))
    parser.add_argument("--checkpoint", default=CONVERTED_CKPT)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK)
    # A drafted token only matters through its argmax, so exact id agreement is
    # the operative criterion; the probability delta is reported for context.
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    prompt_ids, tokenizer = _prompt_ids(args.reference)
    logger.info(f"prompt: {prompt_ids.shape[1]} tokens, block_size={args.block_size}")

    anchor, reference_probs = reference_block(
        args.reference, prompt_ids, args.block_size
    )
    ours_probs = our_block(args.checkpoint, prompt_ids, anchor, args.block_size)

    if reference_probs.shape != ours_probs.shape:
        raise SystemExit(
            f"EQUIVALENCE GATE FAILED: shape {tuple(ours_probs.shape)} != "
            f"reference {tuple(reference_probs.shape)}"
        )
    reference_ids = reference_probs.argmax(-1)
    our_ids = ours_probs.argmax(-1)
    agree = int((reference_ids == our_ids).sum())
    drafted = reference_ids.numel()
    delta = (reference_probs - ours_probs).abs().max().item()

    logger.info(f"anchor token: {tokenizer.decode(anchor[0])!r}")
    logger.info(f"reference draft: {tokenizer.decode(reference_ids[0])!r}")
    logger.info(f"our draft:       {tokenizer.decode(our_ids[0])!r}")
    logger.info(f"token agreement: {agree}/{drafted}   max |delta prob| = {delta:.3e}")

    if agree != drafted:
        first = int((reference_ids != our_ids).nonzero()[0, 1])
        raise SystemExit(
            f"EQUIVALENCE GATE FAILED: {drafted - agree}/{drafted} drafted tokens "
            f"differ, first at block position {first} "
            f"(reference={int(reference_ids[0, first])}, ours={int(our_ids[0, first])})"
        )
    if delta > args.tolerance:
        raise SystemExit(
            f"EQUIVALENCE GATE FAILED: ids agree but max |delta prob| = {delta:.3e} "
            f"exceeds tolerance {args.tolerance:.1e}"
        )
    logger.info("EQUIVALENCE GATE PASSED — our decode geometry matches the reference")


if __name__ == "__main__":
    main()
