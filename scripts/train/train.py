import logging
import os
import time
import traceback
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from rich import get_console
from rich.table import Column, Table

from estuary.clay.data import ClayEstuaryDataModule
from estuary.model.config import EstuaryConfig, ModelType
from estuary.model.data import EstuaryDataModule, calc_class_weights
from estuary.model.module import EstuaryModule
from estuary.util.my_logging import setup_logger

logger = logging.getLogger(__name__)


def main() -> None:
    """
    trains model
    """
    base_conf = OmegaConf.structured(EstuaryConfig)
    cli_conf = OmegaConf.from_cli()
    # cli_conf = OmegaConf.load("train.yaml")
    assert isinstance(cli_conf, DictConfig)

    conf: EstuaryConfig = OmegaConf.merge(base_conf, cli_conf)  # type: ignore

    seed_everything(conf.seed, workers=True)

    # ------------- Run directory (timestamp-based) [C7] -------------
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

    # # sub-dirs Lightning expects
    # (model_dir / "wandb").mkdir(exist_ok=True)

    # Setup logger
    setup_logger(model_dir, log_file_name)

    logger.info(f"Saving results to {model_dir}")
    logger.info(f"Training classification model with classes {conf.classes}")

    # Add dynamic class weights
    if conf.use_class_weights:
        conf.class_weights = calc_class_weights(conf)

    # Load model and data
    model = EstuaryModule(conf)
    if conf.model_type == ModelType.CLAY:
        datamodule = ClayEstuaryDataModule(conf)
    else:
        datamodule = EstuaryDataModule(conf)

    # wandb_logger = WandbLogger(log_model="all", project=conf.project, save_dir=model_dir)
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
    # early_stop = EarlyStopping(
    #     monitor=conf.monitor_metric,
    #     mode=conf.monitor_mode,
    #     patience=conf.patience,
    # )

    # bs_finder = BatchSizeFinder()
    # device_stats = DeviceStatsMonitor()

    assert len(conf.devices) > 0
    devices = list(map(int, conf.devices)) if len(conf.devices) > 1 else conf.devices[0]

    if torch.backends.mps.is_available() and conf.accelerator in ["mps", "auto"]:
        torch._dynamo.config.suppress_errors = True

    trainer = Trainer(
        max_epochs=conf.epochs,
        default_root_dir=model_dir,
        precision=conf.precision,  # type: ignore
        logger=[tensorboard_logger, csv_logger],  # wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],  # early_stop, bs_finder, device_stats
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
        OmegaConf.save(config=cli_conf, f=model_dir / "cli_diff.yaml")
        OmegaConf.save(config=conf, f=model_dir / "conf_full.yaml")
        # trainer.world_size available after instantiation
        assert conf.world_size == trainer.world_size
        # wandb_logger.log_hyperparams(dict(conf))  # type: ignore

    # Train!
    trainer.fit(model=model, datamodule=datamodule)

    # Test!
    results = trainer.test(model=model, datamodule=datamodule, ckpt_path="best", verbose=False)[0]
    columns = [
        Column("Metric", justify="center", style="cyan", width=20),
        Column("Value", justify="center", style="magenta", width=9),
    ]
    table = Table(*columns)
    for metric, v in results.items():
        table.add_row(metric.split("/")[-1], f"{v:.3f}")
    get_console().print(table)

    logger.info("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        traceback.print_exc()
