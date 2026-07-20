from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel

from src.data import build_dataloaders, quiet_download_logs
from src.models.factory import build_lit


def configure_training_device_placement(cfg: DictConfig) -> None:
    """Use Lightning DDP, not Accelerate inference dispatch, for training."""
    device_map = cfg.model.backbone.get("device_map")
    if device_map is not None:
        logger.warning(
            "Ignoring model.backbone.device_map={} for training; Lightning "
            "will place a full model replica on each DDP rank.",
            device_map,
        )
        cfg.model.backbone.device_map = None


class CurrentStreamDDPStrategy(DDPStrategy):
    """Initialize DDP on the stream used by Lightning forwards.

    Lightning 2.6 wraps DDP in a temporary CUDA stream. PyTorch 2.12 then
    observes the DDP-created AccumulateGrad nodes on that stream while the
    training forward runs on the default stream.
    """

    def _setup_model(self, model):
        return DistributedDataParallel(
            module=model,
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )


def configure_ddp_strategy(trainer_kwargs: dict) -> None:
    """Use stream-consistent DDP for explicit or automatic multi-GPU DDP."""
    strategy = trainer_kwargs.get("strategy", "auto")
    if strategy == "ddp":
        trainer_kwargs["strategy"] = CurrentStreamDDPStrategy()
        return
    if strategy != "auto" or trainer_kwargs.get("accelerator", "auto") not in {"auto", "gpu", "cuda"}:
        return

    devices = trainer_kwargs.get("devices", "auto")
    if devices == "auto":
        count = torch.cuda.device_count()
    elif isinstance(devices, int) and devices < 0:
        count = torch.cuda.device_count()
    elif isinstance(devices, (list, tuple)):
        count = len(devices)
    else:
        count = int(devices)
    if count > 1:
        trainer_kwargs["strategy"] = CurrentStreamDDPStrategy()


class ReshuffleStreamingData(L.Callback):
    """Multi-epoch training over a streaming dataset.

    Every new Trainer epoch re-opens the stream from the start; without this
    hook each repetition would replay the SAME sample order. ``set_epoch``
    reseeds the shuffle buffer, so repetitions see the same data in a new
    order — the streaming equivalent of an epoch. The val slice is split off
    before the shuffle (see build_dataloaders), so its membership never
    changes.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        ds = getattr(trainer.train_dataloader, "dataset", None)
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(trainer.current_epoch)


class FinalCheckpoint(L.Callback):
    """Always persist the terminal training state to one predictable path.

    Lightning's ``ModelCheckpoint(save_last=True)`` can leave ``last.ckpt`` at
    an earlier periodic step. This callback explicitly saves after the final
    optimizer step and also makes a best effort when training raises.
    """

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self._saved = False

    def _save(self, trainer, *, reason: str):
        if self._saved or trainer.fast_dev_run:
            return
        try:
            trainer.save_checkpoint(self.path)
        except Exception:
            logger.exception(f"failed to save {reason} checkpoint to {self.path}")
            if reason == "final":
                raise
        else:
            self._saved = True
            if trainer.is_global_zero:
                logger.info(f"{reason} checkpoint saved -> {self.path}")

    def on_train_end(self, trainer, pl_module):
        self._save(trainer, reason="final")

    def on_exception(self, trainer, pl_module, exception):
        self._save(trainer, reason="exception")


def build_checkpoint_callbacks(cfg: DictConfig, *, has_validation: bool):
    """Independent recovery, model-selection, and terminal checkpoints."""
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    periodic_steps = int(cfg.train.get("checkpoint_every_n_steps", 0))
    callbacks = []
    if periodic_steps > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename=cfg.train.get("checkpoint_name", "flowdraft-{step:07d}"),
                monitor=None,
                save_top_k=-1,
                save_last=False,
                every_n_train_steps=periodic_steps,
                every_n_epochs=0,
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False,
            )
        )

    if has_validation:
        monitor = cfg.train.get("monitor", "val/tpf")
        mode = cfg.train.get("monitor_mode", "max")
        monitor_slug = str(monitor).replace("/", "-")
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename=cfg.train.get(
                    "best_checkpoint_name", f"best-{monitor_slug}-{{step:07d}}"
                ),
                monitor=monitor,
                mode=mode,
                save_top_k=int(cfg.train.get("checkpoint_save_top_k", 2)),
                save_last=False,
                every_n_train_steps=0,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False,
            )
        )

    final_name = cfg.train.get("final_checkpoint_name", "last.ckpt")
    callbacks.append(FinalCheckpoint(str(output_dir / final_name)))
    return callbacks


def build_loggers(cfg: DictConfig):
    """Keep local TensorBoard logs and optionally mirror every metric to W&B."""
    loggers = [TensorBoardLogger(save_dir=cfg.output_dir)]
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return loggers

    # Import lazily so local/TensorBoard-only runs do not initialize W&B.
    from lightning.pytorch.loggers import WandbLogger

    kwargs = {
        "project": wandb_cfg.get("project", "flowdraft"),
        "save_dir": cfg.output_dir,
        "offline": wandb_cfg.get("offline", False),
        "log_model": False,
    }
    for key in ("entity", "name", "group"):
        value = wandb_cfg.get(key)
        if value is not None:
            kwargs[key] = value
    tags = wandb_cfg.get("tags")
    if tags:
        kwargs["tags"] = list(tags)
    loggers.append(WandbLogger(**kwargs))
    return loggers


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    quiet_download_logs()
    L.seed_everything(cfg.seed, workers=True)
    configure_training_device_placement(cfg)
    model = build_lit(
        cfg,
        variant=cfg.train.get("variant", "flowdraft"),
        # A weights-only warm start must keep this run's optimizer, monitor,
        # checkpoint names, and CLI overrides.
        restore_train_config=False,
    )
    train_loader, val_loader = build_dataloaders(cfg, model.tokenizer, model.df_processor)

    if (
        val_loader is not None
        and cfg.train.get("monitor", "val/tpf") == "val/tpf"
        and cfg.train.get("val_decode_prompts", 0) <= 0
    ):
        raise ValueError(
            "train.monitor=val/tpf requires train.val_decode_prompts > 0"
        )
    callbacks = [
        *build_checkpoint_callbacks(cfg, has_validation=val_loader is not None),
        LearningRateMonitor(logging_interval="step"),
        ReshuffleStreamingData(),
    ]
    # Stop when the validation loss starts GROWING (patience = how many
    # validations in a row it may fail to improve); 0 disables.
    patience = cfg.train.get("early_stop_patience", 0)
    if patience > 0 and val_loader is not None:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=patience))
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    configure_ddp_strategy(trainer_kwargs)
    trainer = L.Trainer(
        callbacks=callbacks,
        default_root_dir=cfg.output_dir,
        logger=build_loggers(cfg),
        **trainer_kwargs,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.get("resume_from_checkpoint"),
    )


if __name__ == "__main__":
    main()
