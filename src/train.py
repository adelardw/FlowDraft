import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.data import build_dataloaders, quiet_download_logs
from src.models.factory import build_lit


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    quiet_download_logs()
    L.seed_everything(cfg.seed, workers=True)

    model = build_lit(cfg, variant=cfg.train.get("variant", "fixed"))
    train_loader, val_loader = build_dataloaders(cfg, model.tokenizer, model.df_processor)

    # Checkpoint selection: val/tpf (the target metric; needs val_decode_prompts>0)
    # or val/loss; falls back to train/loss when there is no val loader.
    if val_loader is None:
        monitor, mode = "train/loss", "min"
    elif cfg.train.get("val_decode_prompts", 0) > 0:
        monitor = cfg.train.get("monitor", "val/tpf")
        mode = cfg.train.get("monitor_mode", "max")
    else:
        monitor, mode = "val/loss", "min"
    callbacks = [
        # on_save_checkpoint strips the frozen backbone; the file still
        # carries the DF head + its Adam moments. Keep the best and the last.
        ModelCheckpoint(
            dirpath=cfg.output_dir,
            filename=cfg.train.get("checkpoint_name", "flowdraft-{step:07d}"),
            monitor=monitor,
            mode=mode,
            save_top_k=2,
            save_last=True,
            every_n_train_steps=cfg.train.get("checkpoint_every_n_steps", 1000),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    # Stop when the validation loss starts GROWING (patience = how many
    # validations in a row it may fail to improve); 0 disables.
    patience = cfg.train.get("early_stop_patience", 0)
    if patience > 0 and val_loader is not None:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=patience))
    trainer = L.Trainer(
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
        **OmegaConf.to_container(cfg.trainer, resolve=True),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
