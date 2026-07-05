import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.data import build_dataloaders
from src.models.factory import build_lit


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    model = build_lit(cfg, variant=cfg.train.get("variant", "fixed"))
    train_loader, val_loader = build_dataloaders(cfg, model.tokenizer, model.df_processor)

    callbacks = [
        # The DF head is ~1.7 GB (on_save_checkpoint strips the frozen
        # backbone); keep the best and the last.
        ModelCheckpoint(
            dirpath=cfg.output_dir,
            filename="flowdraft-{step:07d}",
            monitor="val/loss" if val_loader is not None else "train/loss",
            save_top_k=2,
            save_last=True,
            every_n_train_steps=cfg.train.get("checkpoint_every_n_steps", 1000),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = L.Trainer(
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
        **OmegaConf.to_container(cfg.trainer, resolve=True),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
