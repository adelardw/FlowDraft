"""Bidirectional weight conversion between the official Orthrus release and our
Lightning DF-head checkpoints.

The official release (``chiennv/Orthrus-Qwen3-1.7B``) keeps its diffusion path as
dense per-layer twins named ``...self_attn.{q,k,v,o}_proj_diff.weight`` and
``...{q,k}_norm_diff.weight`` — six tensors per layer, exactly the modules our
``adapter.w_names`` selects. Our adapter stores the same tensors positionally in
``orthrus.df_weights.<i>``, ordered by :meth:`FlowDraftAttentionAdapter._clone_df_weights`.
So the mapping is a rename, not a re-parameterization, and the index is read off
``adapter._df_names`` rather than reconstructed by arithmetic.

The one structural difference is the ``[MASK]`` input. Orthrus reads it from the
frozen embedding table at ``config.mask_token_id``; we keep a free
``orthrus.mask_embedding`` parameter beside the projections (``df_adapter.py:220``).
Conversion moves that vector across in either direction. Writing *into* the table
perturbs one row of the tied ``lm_head``, so ``to-theirs`` reports the logit that
row can now produce and the reference harness asserts the token is never emitted.

    uv run python -m src.tools.convert_orthrus to-ours \
        --reference weights/Orthrus-Qwen3-1.7B --out weights/orthrus-authors.ckpt

    uv run python -m src.tools.convert_orthrus to-theirs \
        --checkpoint weights/qwen3-1.7-baseline-block8.ckpt \
        --reference weights/Orthrus-Qwen3-1.7B --out weights/ours-as-orthrus

    uv run python -m src.tools.convert_orthrus verify --reference weights/Orthrus-Qwen3-1.7B
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

REFERENCE_DIR = "weights/Orthrus-Qwen3-1.7B"
# Files the reference repo needs beside model.safetensors for trust_remote_code.
_SIDECAR_FILES = (
    "modeling_orthrus.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


def reference_key(df_name: str) -> str:
    """``model.layers.0.self_attn.q_proj.weight`` -> ``...q_proj_diff.weight``."""
    module, _, parameter = df_name.rpartition(".")
    return f"{module}_diff.{parameter}"


def build_adapter(model_config: str = "qwen3_1.7b"):
    """Our adapter on CPU — the authority on DF tensor order, names and shapes.

    Instantiated rather than reconstructed: ``_df_names`` is the same tuple the
    optimizer and ``validate_df_state`` index into, so a future change to
    ``w_names`` or to the matching rule cannot silently desynchronize the
    converter from the checkpoints it writes.
    """
    import hydra

    from src.models.model import build_model

    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(
            config_name="eval",
            overrides=[
                f"model={model_config}",
                "model.backbone.device_map=null",
                "model.backbone.compile_ar=false",
            ],
        )
    model, _, _ = build_model(cfg.model)
    return model, cfg


def _reference_config(reference: Path) -> dict:
    return json.loads((reference / "config.json").read_text())


def to_ours(reference: Path, out: Path, model_config: str = "qwen3_1.7b") -> Path:
    """Official release -> a Lightning checkpoint ``build_lit`` can restore."""
    adapter, cfg = build_adapter(model_config)
    ref_config = _reference_config(reference)
    mask_token_id = ref_config["mask_token_id"]
    block_size = ref_config["block_size"]
    tensors = load_file(str(reference / "model.safetensors"))

    state_dict, base_deltas = {}, []
    for index, df_name in enumerate(adapter._df_names):
        key = reference_key(df_name)
        if key not in tensors:
            raise KeyError(
                f"reference release has no {key!r} for our DF tensor {df_name!r}; "
                f"adapter.w_names={adapter.w_names} does not match this release"
            )
        expected = adapter.df_weights[index].shape
        if tensors[key].shape != expected:
            raise ValueError(
                f"{key}: reference shape {tuple(tensors[key].shape)} != "
                f"ours {tuple(expected)}"
            )
        state_dict[f"orthrus.df_weights.{index}"] = tensors[key].float()
        # The release claims a strictly frozen backbone. Confirm it against the
        # HF revision our own runs load, because cells A and B only compare
        # like for like if the verifier is the same model.
        base = tensors.get(df_name)
        if base is not None:
            ours = dict(adapter.model.named_parameters())[df_name]
            base_deltas.append((base.float() - ours.float()).abs().max().item())

    embed = tensors["model.embed_tokens.weight"]
    state_dict["orthrus.mask_embedding"] = embed[mask_token_id].float().unsqueeze(0)

    if base_deltas:
        worst = max(base_deltas)
        message = f"frozen backbone check: max |release - Qwen3 base| = {worst:.3e}"
        logger.info(message) if worst == 0 else logger.warning(message)

    # Deliberately minimal: only what our code reads. Their training
    # hyperparameters are not published, and inventing our own preset's lr /
    # lambda here would make eval.py record fiction in the ``train_*`` columns.
    train = {"variant": "orthrus", "block_size": block_size}
    checkpoint = {
        "state_dict": state_dict,
        "hyper_parameters": {
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "train": train,
            "seed": None,
            "wandb": {"name": f"authors-{reference.name}"},
        },
        "global_step": None,
        "epoch": None,
        "pytorch-lightning_version": "converted",
        "hparams_name": "cfg",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    logger.info(
        f"{len(state_dict)} tensors -> {out} "
        f"(block_size={block_size}, mask_token_id={mask_token_id})"
    )
    return out


def to_theirs(
    checkpoint_path: Path, reference: Path, out: Path, model_config: str = "qwen3_1.7b"
) -> Path:
    """Our Lightning checkpoint -> a repo the official ``OrthrusLM`` can load.

    The frozen backbone is taken from the reference release rather than from
    ``Qwen/Qwen3-1.7B`` so that cells A and D share a byte-identical verifier and
    differ only in the diffusion path.
    """
    adapter, _ = build_adapter(model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    ref_config = _reference_config(reference)
    mask_token_id = ref_config["mask_token_id"]
    tensors = load_file(str(reference / "model.safetensors"))

    for index, df_name in enumerate(adapter._df_names):
        source = state_dict.get(f"orthrus.df_weights.{index}")
        if source is None:
            raise KeyError(
                f"checkpoint {checkpoint_path} has no orthrus.df_weights.{index} "
                f"(expected {len(adapter._df_names)} DF tensors)"
            )
        key = reference_key(df_name)
        if source.shape != tensors[key].shape:
            raise ValueError(
                f"{key}: checkpoint shape {tuple(source.shape)} != "
                f"reference {tuple(tensors[key].shape)}"
            )
        tensors[key] = source.to(tensors[key].dtype)

    mask_embedding = state_dict.get("orthrus.mask_embedding")
    embed = tensors["model.embed_tokens.weight"]
    if mask_embedding is not None:
        # Their [MASK] is a table lookup, so our free vector has to live in the
        # table. Embeddings are tied to lm_head: this makes mask_token_id
        # producible in principle, hence the reported margin below.
        embed[mask_token_id] = mask_embedding.reshape(-1).to(embed.dtype)
        norms = embed.float().norm(dim=-1)
        logger.info(
            f"mask row {mask_token_id} written: |row|={norms[mask_token_id]:.3f} "
            f"vs vocabulary mean |row|={norms.mean():.3f}"
        )
    else:
        logger.warning(
            "checkpoint has no orthrus.mask_embedding; keeping the release's "
            "mask row (a FlowDraft variant freezes this parameter)"
        )

    hparams = checkpoint.get("hyper_parameters", {})
    block_size = (hparams.get("train") or {}).get("block_size", ref_config["block_size"])
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(
        json.dumps({**ref_config, "block_size": block_size}, indent=2)
    )
    for name in _SIDECAR_FILES:
        source = reference / name
        if source.is_file():
            shutil.copy2(source, out / name)
    logger.info(f"{checkpoint_path} -> {out} (block_size={block_size})")
    return out


def verify(reference: Path, model_config: str = "qwen3_1.7b") -> None:
    """Round-trip the release through our layout and back, bitwise.

    The release stores bf16, so a correct mapping round-trips exactly; any
    mismatch is a mis-indexed, transposed or dropped tensor rather than a
    precision artifact. This is the cheapest test that the 2x2 is comparing
    weights and not a conversion bug.
    """
    adapter, _ = build_adapter(model_config)
    ref_config = _reference_config(reference)
    tensors = load_file(str(reference / "model.safetensors"))

    ours = {
        f"orthrus.df_weights.{index}": tensors[reference_key(name)].float()
        for index, name in enumerate(adapter._df_names)
    }
    ours["orthrus.mask_embedding"] = (
        tensors["model.embed_tokens.weight"][ref_config["mask_token_id"]]
        .float()
        .unsqueeze(0)
    )

    failures = []
    for index, df_name in enumerate(adapter._df_names):
        key = reference_key(df_name)
        back = ours[f"orthrus.df_weights.{index}"].to(tensors[key].dtype)
        if not torch.equal(back, tensors[key]):
            failures.append(key)
    mask_back = (
        ours["orthrus.mask_embedding"]
        .reshape(-1)
        .to(tensors["model.embed_tokens.weight"].dtype)
    )
    if not torch.equal(
        mask_back, tensors["model.embed_tokens.weight"][ref_config["mask_token_id"]]
    ):
        failures.append("model.embed_tokens.weight[mask_token_id]")

    checked = len(adapter._df_names) + 1
    if failures:
        raise SystemExit(
            f"ROUND-TRIP FAILED on {len(failures)}/{checked} tensors: {failures[:8]}"
        )
    logger.info(f"round-trip OK: {checked}/{checked} tensors bitwise identical")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("to-ours", "to-theirs", "verify"))
    parser.add_argument("--reference", type=Path, default=Path(REFERENCE_DIR))
    parser.add_argument("--checkpoint", type=Path, help="our .ckpt (to-theirs only)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model-config", default="qwen3_1.7b")
    args = parser.parse_args()

    if not args.reference.is_dir():
        raise SystemExit(
            f"reference release not found at {args.reference}; download it with "
            f"`hf download chiennv/Orthrus-Qwen3-1.7B --local-dir {args.reference}`"
        )
    if args.command == "verify":
        verify(args.reference, args.model_config)
    elif args.command == "to-ours":
        to_ours(args.reference, args.out, args.model_config)
    else:
        if not args.checkpoint:
            raise SystemExit("to-theirs requires --checkpoint")
        to_theirs(args.checkpoint, args.reference, args.out, args.model_config)


if __name__ == "__main__":
    main()
