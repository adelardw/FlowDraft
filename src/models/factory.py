import torch
from loguru import logger
from omegaconf import DictConfig


def build_lit(cfg: DictConfig, variant: str | None = None):
    """Build a FlowDraft or Orthrus module by variant name.

    Lives apart from ``model.py`` on purpose: lit modules import
    ``build_model`` from there, so a factory of lit modules inside the same
    file would be one step from an import cycle.

    Canonical variants are ``flowdraft``, ``flowdraft_block_wise``,
    ``orthrus``, and ``orthrus_block_wise``. The former names remain aliases
    so existing commands and checkpoints keep loading.
    Loads the DF head from ``cfg.checkpoint`` when set — the frozen backbone
    always comes from HF via ``build_model``.
    """
    variant = variant or cfg.get("variant", "flowdraft")
    aliases = {
        "fixed": "flowdraft",
        "block_wise": "flowdraft_block_wise",
        "baseline": "orthrus",
        "baseline_block_wise": "orthrus_block_wise",
    }
    canonical_variant = aliases.get(variant, variant)
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
            f"unknown variant='{variant}' (flowdraft | flowdraft_block_wise | "
            "orthrus | orthrus_block_wise)"
        )
    model = Module(cfg)
    if cfg.get("checkpoint"):
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=False)  # ckpt holds the DF head only
        logger.info(f"DF head loaded from {cfg.checkpoint}")
    return model
