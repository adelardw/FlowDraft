import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # from src.models.model import build_model
    # backbone, tokenizer = build_model(cfg.model)
    # ... run lossless decoding + verify, measure acceptance length / TPF / throughput


if __name__ == "__main__":
    main()
