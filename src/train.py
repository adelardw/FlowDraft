import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # from src.models.model import build_model
    # backbone, tokenizer = build_model(cfg.model)
    # ... train the flow-map drafter (dual distillation)


if __name__ == "__main__":
    main()
