from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base.df_adapter import OrthrusAttentionAdapter
from src.preprocessor.df_processor import DiffusionProcessor


def build_model(cfg: DictConfig):
    """Build the Orthrus model: frozen AR backbone + trainable diffusion head.

    Args:
        cfg: the `model` config node (see ``configs/model/*.yaml``) with
            ``backbone``, ``tokenizer`` and ``adapter`` sub-nodes.

    Returns:
        ``(model, tokenizer, processor)`` where
          - ``model`` is the backbone wrapped in :class:`OrthrusAttentionAdapter`
            (frozen AR path + trainable df path, routed by ``forward(use_df=...)``),
          - ``tokenizer`` is the HF tokenizer (AR path inputs),
          - ``processor`` builds the simplex/one-hot endpoints for the df path.
    """
    backbone_kwargs = OmegaConf.to_container(cfg.backbone, resolve=True)
    tokenizer_kwargs = OmegaConf.to_container(cfg.tokenizer, resolve=True)

    backbone = AutoModelForCausalLM.from_pretrained(**backbone_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)

    # Frozen AR backbone — this is what makes the output provably lossless.
    backbone.requires_grad_(False)
    backbone.eval()

    processor = DiffusionProcessor.from_model(tokenizer, backbone)
    model = OrthrusAttentionAdapter(backbone, w_names=list(cfg.adapter.w_names))

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Orthrus '{cfg.name}': {n_total / 1e9:.2f}B total, "
        f"{n_train / 1e6:.1f}M trainable (df head over {model.n_dual} projections)"
    )
    return model, tokenizer, processor


if __name__ == "__main__":
    # Debug: compose the config and print it resolved. Uncomment build_model to
    # actually download + load the backbone (gated model — needs HF auth).
    import hydra

    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(config_name="train")
    print(OmegaConf.to_yaml(cfg.model, resolve=True))
    model, tokenizer, processor = build_model(cfg.model)
