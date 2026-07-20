from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base.df_adapter import FlowDraftAttentionAdapter
from src.preprocessor.df_processor import DiffusionProcessor


def build_model(cfg: DictConfig):
    """Build the shared FlowDraft/Orthrus backbone and diffusion head.

    Args:
        cfg: the `model` config node (see ``configs/model/*.yaml``) with
            ``backbone``, ``tokenizer`` and ``adapter`` sub-nodes.

    Returns:
        ``(model, tokenizer, processor)`` where
          - ``model`` is the backbone wrapped in :class:`FlowDraftAttentionAdapter`
            (frozen AR path + trainable df path, routed by ``forward(use_df=...)``),
          - ``tokenizer`` is the HF tokenizer (AR path inputs),
          - ``processor`` builds the simplex/one-hot endpoints for the df path.
    """
    backbone_kwargs = OmegaConf.to_container(cfg.backbone, resolve=True)
    # These are FlowDraft runtime options, not Hugging Face
    # ``from_pretrained`` arguments. Compile only the frozen AR branch: the
    # DF branch uses ``torch.func.functional_call`` to substitute trainable
    # diffusion-attention twins and must keep calling the original module.
    compile_ar = backbone_kwargs.pop("compile_ar", False)
    compile_mode = backbone_kwargs.pop("compile_mode", "default")
    compile_dynamic = backbone_kwargs.pop("compile_dynamic", False)
    gradient_checkpointing = backbone_kwargs.pop("gradient_checkpointing", False)
    tokenizer_kwargs = OmegaConf.to_container(cfg.tokenizer, resolve=True)

    backbone = AutoModelForCausalLM.from_pretrained(**backbone_kwargs)
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    if tokenizer.pad_token is None:
        # Llama-family tokenizers ship without a pad token, while
        # DiffusionProcessor pads by default — the first real batched
        # tokenization would crash otherwise.
        tokenizer.pad_token = tokenizer.eos_token

    # Frozen AR backbone — this is what makes the output provably lossless.
    backbone.requires_grad_(False)
    backbone.eval()

    df_processor = DiffusionProcessor.from_model(tokenizer, backbone)
    model = FlowDraftAttentionAdapter(backbone, w_names=list(cfg.adapter.w_names))
    if compile_ar:
        model.enable_ar_compile(mode=compile_mode, dynamic=compile_dynamic)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"FlowDraft '{cfg.name}': {n_total / 1e9:.2f}B total, "
        f"{n_train / 1e6:.1f}M trainable (df head over {model.n_dual} projections)"
    )
    return model, tokenizer, df_processor


if __name__ == "__main__":
    # Debug: compose the config and print it resolved. Uncomment build_model to
    # actually download + load the backbone (gated model — needs HF auth).
    import hydra

    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(config_name="train")
    print(OmegaConf.to_yaml(cfg.model, resolve=True))
    model, tokenizer, processor = build_model(cfg.model)
