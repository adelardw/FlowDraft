import torch
from loguru import logger
from omegaconf import DictConfig


def build_lit(cfg: DictConfig, variant: str | None = None):
    """FlowMapOrthrus family by variant name, checkpoint included.

    Lives apart from ``model.py`` on purpose: lit modules import
    ``build_model`` from there, so a factory of lit modules inside the same
    file would be one step from an import cycle.

    ``variant``: fixed | block_wise | baseline (default from ``cfg.variant``).
    Loads the DF head from ``cfg.checkpoint`` when set — the frozen backbone
    always comes from HF via ``build_model``.
    """
    variant = variant or cfg.get("variant", "fixed")
    if variant == "baseline":
        from src.models.lit_orthrus_baseline import FlowMapOrthrusBaseline as Module
    elif variant == "block_wise":
        from src.models.lit_orthrus_block_wise import FlowMapOrthrusBlockWise as Module
    elif variant == "fixed":
        from src.models.lit_orthrus import FlowMapOrthrus as Module
    else:
        raise ValueError(f"unknown variant='{variant}' (fixed | block_wise | baseline)")
    model = Module(cfg)
    if cfg.get("checkpoint"):
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=False)  # ckpt holds the DF head only
        logger.info(f"DF head loaded from {cfg.checkpoint}")
    return model
