from collections.abc import Mapping
from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict


_VARIANT_ALIASES = {
    "fixed": "flowdraft",
    "block_wise": "flowdraft_block_wise",
    "baseline": "orthrus",
    "baseline_block_wise": "orthrus_block_wise",
}
_RUNTIME_BACKBONE_KEYS = {
    "attn_implementation",
    "compile_ar",
    "compile_dynamic",
    "compile_mode",
    "device_map",
    "dtype",
    "low_cpu_mem_usage",
}


def _canonical_variant(variant: str | None) -> str:
    variant = variant or "flowdraft"
    return _VARIANT_ALIASES.get(variant, variant)


def _checkpoint_hparams(checkpoint: Mapping) -> Mapping:
    hparams = checkpoint.get("hyper_parameters", {})
    if not isinstance(hparams, Mapping):
        return {}
    # Also accept checkpoints produced by ``save_hyperparameters("cfg")``.
    nested = hparams.get("cfg")
    return nested if isinstance(nested, Mapping) else hparams


def _checkpoint_variant(checkpoint: Mapping) -> str | None:
    hparams = _checkpoint_hparams(checkpoint)
    train = hparams.get("train", {})
    if isinstance(train, Mapping) and train.get("variant"):
        return _canonical_variant(train["variant"])
    if hparams.get("variant"):
        return _canonical_variant(hparams["variant"])
    return None


def _load_checkpoint(path: str) -> tuple[dict, Path]:
    resolved = Path(to_absolute_path(path)).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint does not exist or is not a file: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must contain a mapping, got {type(checkpoint).__name__}")
    if not isinstance(checkpoint.get("state_dict"), Mapping):
        raise KeyError(f"checkpoint has no Lightning 'state_dict': {resolved}")
    return checkpoint, resolved


def _restore_checkpoint_config(
    cfg: DictConfig, checkpoint: Mapping, *, restore_train: bool = True
) -> None:
    """Restore architecture/training metadata without changing eval placement.

    Training under DDP records ``device_map=null``. Evaluation must retain its
    own device, dtype, attention-kernel, and compile settings, while model ID,
    tokenizer, adapter projections, and train-time knobs come from the file.
    """
    hparams = _checkpoint_hparams(checkpoint)
    saved_model = hparams.get("model")
    saved_train = hparams.get("train")
    with open_dict(cfg):
        if isinstance(saved_model, Mapping):
            runtime_backbone = {
                key: cfg.model.backbone.get(key)
                for key in _RUNTIME_BACKBONE_KEYS
                if cfg.get("model") and cfg.model.get("backbone")
                and cfg.model.backbone.get(key) is not None
            }
            restored_model = OmegaConf.create(saved_model)
            if restored_model.get("backbone") is not None:
                for key, value in runtime_backbone.items():
                    restored_model.backbone[key] = value
            cfg.model = restored_model
        if restore_train and isinstance(saved_train, Mapping):
            cfg.train = OmegaConf.create(saved_train)


def validate_df_state(model, state: Mapping, source: str) -> None:
    """Require every trainable DF tensor with its expected shape."""
    model_state = model.state_dict()
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    present = set(state)
    unknown = present - set(model_state)
    missing = required - present
    mismatched = {
        name: (tuple(state[name].shape), tuple(model_state[name].shape))
        for name in required & present
        if state[name].shape != model_state[name].shape
    }
    if unknown or missing or mismatched:
        details = []
        if missing:
            details.append(f"missing trainable tensors={sorted(missing)}")
        if unknown:
            details.append(f"unknown tensors={sorted(unknown)}")
        if mismatched:
            details.append(f"shape mismatches={mismatched}")
        raise RuntimeError(f"incompatible DF checkpoint {source}: " + "; ".join(details))


def _load_df_state(model, checkpoint: Mapping, checkpoint_path: Path) -> None:
    """Strictly restore every trainable DF tensor and no frozen backbone."""
    state = checkpoint["state_dict"]
    validate_df_state(model, state, str(checkpoint_path))
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    # Older/full Lightning files may include valid frozen-backbone tensors.
    # Deliberately ignore them: the frozen verifier must come from the model
    # revision recorded in checkpoint metadata, not from an accidental copy.
    trainable_state = {name: state[name] for name in required}
    model.load_state_dict(trainable_state, strict=False)
    model.checkpoint_path = str(checkpoint_path)
    model.checkpoint_global_step = checkpoint.get("global_step")
    model.checkpoint_epoch = checkpoint.get("epoch")
    logger.info(
        f"loaded {len(trainable_state)} DF tensors from {checkpoint_path} "
        f"(step={model.checkpoint_global_step}, epoch={model.checkpoint_epoch})"
    )


def build_lit(
    cfg: DictConfig,
    variant: str | None = None,
    *,
    restore_train_config: bool = True,
):
    """Build FlowDraft/Orthrus and restore a checkpoint when configured.

    Evaluation checkpoints restore their saved model architecture, training
    parameters, variant, and complete trainable DF state. Set
    ``checkpoint_config=false`` only to intentionally ignore saved metadata.
    """
    checkpoint = None
    checkpoint_path = None
    requested_variant = variant or cfg.get("variant")
    if cfg.get("checkpoint"):
        checkpoint, checkpoint_path = _load_checkpoint(str(cfg.checkpoint))
        saved_variant = _checkpoint_variant(checkpoint)
        if cfg.get("checkpoint_config", True):
            hparams = _checkpoint_hparams(checkpoint)
            if not isinstance(hparams.get("model"), Mapping):
                raise ValueError(
                    "checkpoint has no saved model configuration; provide the "
                    "matching model/variant explicitly and set checkpoint_config=false"
                )
            if saved_variant is None and requested_variant is None:
                raise ValueError(
                    "checkpoint has no saved variant; set variant explicitly or "
                    "set checkpoint_config=false"
                )
            if requested_variant and saved_variant:
                if _canonical_variant(requested_variant) != saved_variant:
                    raise ValueError(
                        f"requested variant={requested_variant!r}, but checkpoint "
                        f"was trained as {saved_variant!r}"
                    )
            _restore_checkpoint_config(
                cfg, checkpoint, restore_train=restore_train_config
            )
            requested_variant = saved_variant or requested_variant

    canonical_variant = _canonical_variant(requested_variant)
    with open_dict(cfg):
        cfg.variant = canonical_variant

    if canonical_variant == "orthrus":
        from src.models.orthrus import Orthrus as Module
    elif canonical_variant == "orthrus_block_wise":
        from src.models.orthrus_block_wise import OrthrusBlockWise as Module
    elif canonical_variant == "flowdraft_block_wise":
        from src.models.flowdraft_block_wise import FlowDraftBlockWise as Module
    elif canonical_variant == "flowdraft":
        from src.models.flowdraft import FlowDraft as Module
    else:
        raise ValueError(
            f"unknown variant='{requested_variant}' (flowdraft | flowdraft_block_wise | "
            "orthrus | orthrus_block_wise)"
        )

    model = Module(cfg)
    if checkpoint is not None:
        _load_df_state(model, checkpoint, checkpoint_path)
    return model
