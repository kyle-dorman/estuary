import logging
import os
import time
from pathlib import Path

import click
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf

from estuary.clay.data import EstuaryDataModule
from estuary.clay.module import EstuaryModule
from estuary.util import setup_logger

logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--checkpoint-path", type=click.Path(exists=True), required=True)
@click.option("-r", "--holdout-region", type=str, required=True)
@click.option("-e", "--epochs", type=int, required=True)
def main(
    checkpoint_path: Path,
    holdout_region: str,
    epochs: int,
):
    """
    finetune and hold out a specific region
    """
    checkpoint_path = Path(checkpoint_path)

    # ------------------------------------------------------------------
    # Build a fresh `conf` and load checkpoint weights into that model
    # ------------------------------------------------------------------
    # 1) start from the config stored inside the checkpoint
    model = EstuaryModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        epochs=epochs,
        warmup_epochs=0,
    )
    conf = model.conf
    conf.holdout_region = holdout_region

    # ------------- Run directory (timestamp‑based) [C7] -------------
    rank = os.environ.get("NODE_RANK", None)
    model_training_root = Path(conf.model_training_root) / conf.project / "train"
    model_training_root.mkdir(exist_ok=True, parents=True)
    if rank is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        model_dir = model_training_root / run_id
        model_dir.mkdir()
        log_file_name = "log.log"
    else:
        model_dir = sorted(list(model_training_root.glob("*-*")))[-1]
        log_file_name = f"log_{rank}.log"

    # Setup logger
    setup_logger(logger, model_dir, log_file_name)

    logger.info(f"Saving results to {model_dir}")
    logger.info(f"Resume training for {checkpoint_path}")

    # Load data
    datamodule = EstuaryDataModule(conf)

    tensorboard_logger = TensorBoardLogger(model_dir)
    csv_logger = CSVLogger(model_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir / "checkpoints",
        monitor=conf.monitor_metric,
        mode=conf.monitor_mode,
        every_n_epochs=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(
        monitor=conf.monitor_metric,
        mode=conf.monitor_mode,
        patience=conf.patience,
    )

    assert len(conf.devices) > 0
    devices = list(map(int, conf.devices)) if len(conf.devices) > 1 else conf.devices[0]

    if torch.backends.mps.is_available() and conf.accelerator in ["mps", "auto"]:
        torch._dynamo.config.suppress_errors = True

    trainer = Trainer(
        max_epochs=conf.epochs,
        default_root_dir=model_dir,
        precision=conf.precision,  # type: ignore
        logger=[tensorboard_logger, csv_logger],
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
        deterministic=conf.deterministic,
        devices=devices,
        accelerator=conf.accelerator,
        log_every_n_steps=conf.log_every_n_steps,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        accumulate_grad_batches=conf.grad_accum_steps,
        enable_progress_bar=len(conf.devices) == 1,
    )

    if trainer.is_global_zero:
        # Save configs – full + diff
        OmegaConf.save(config=conf, f=model_dir / "conf_full.yaml")
        # trainer.world_size available after instantiation
        assert conf.world_size == trainer.world_size
        # wandb_logger.log_hyperparams(dict(conf))  # type: ignore

    # Train!
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
