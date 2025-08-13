import logging

import torch
from claymodel.module import ClayMAEModule
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryCalibrationError,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from estuary.clay.classifier import Classifier, ConvDecoder, TransformerDecoder
from estuary.clay.config import EstuaryConfig

logger = logging.getLogger(__name__)


class EstuaryModule(LightningModule):
    def __init__(self, conf: EstuaryConfig):
        self.conf = conf
        self.num_classes = len(conf.classes)

        super().__init__()
        self.save_hyperparameters(conf)

        clay_model = ClayMAEModule.load_from_checkpoint(
            conf.encoder_weights,
            metadata_path=conf.metadata_path,
            strict=False,
            mask_ratio=0.0,
            shuffle=False,
        )
        if conf.freeze_encoder:
            clay_model = clay_model.eval()
        encoder = clay_model.model.encoder
        # Freeze the encoder parameters
        if conf.freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
        assert encoder.dim == conf.encoder_dim
        decoder = ConvDecoder(conf) if conf.decoder_name == "conv" else TransformerDecoder(conf)
        clf = Classifier(encoder, decoder)

        using_mps = torch.backends.mps.is_available() and conf.accelerator in ["mps", "auto"]
        if conf.debug or using_mps:
            self.model = clf
        else:
            self.model = torch.compile(clf)  # type: ignore

        logger.info("Creating metrics")

        metrics_dict = {
            "f1": MulticlassF1Score(num_classes=self.num_classes),
            "precision": MulticlassPrecision(num_classes=self.num_classes),
            "recall": MulticlassRecall(num_classes=self.num_classes),
            "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
            "cohenkappa": MulticlassCohenKappa(num_classes=self.num_classes),
            "auroc": MulticlassAUROC(num_classes=self.num_classes),
        }
        # per-class precision vector
        metrics_dict["precision_perclass"] = MulticlassPrecision(
            num_classes=self.num_classes, average="none"
        )
        metrics = MetricCollection(metrics_dict)

        # metrics_device = "cpu" if using_mps else self.device
        self.train_metrics = metrics.clone(prefix="train/").to(self.device)
        self.train_cm = MulticlassConfusionMatrix(num_classes=self.num_classes).to(self.device)
        self.train_ece = BinaryCalibrationError()
        self.val_metrics = metrics.clone(prefix="val/").to(self.device)
        self.val_cm = MulticlassConfusionMatrix(num_classes=self.num_classes).to(self.device)
        self.val_ece = BinaryCalibrationError()
        self.test_metrics = metrics.clone(prefix="test/").to(self.device)
        self.test_cm = MulticlassConfusionMatrix(num_classes=self.num_classes).to(self.device)
        self.test_ece = BinaryCalibrationError()

        if conf.class_weights is not None:
            weights = torch.tensor(conf.class_weights, device=self.device)
        else:
            weights = None
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=conf.smooth_factor, weight=weights)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(x)

    def step(
        self,
        train_test_val: str,
        batch_target: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        if train_test_val == "train":
            metrics = self.train_metrics
            cm = self.train_cm
            ece = self.train_ece
        elif train_test_val == "test":
            metrics = self.test_metrics
            cm = self.test_cm
            ece = self.val_ece
        else:
            metrics = self.val_metrics
            cm = self.val_cm
            ece = self.test_ece

        batch, target = batch_target
        pred: torch.Tensor = self(batch)

        # Loss
        loss = self.loss_fn(pred, target)

        # Grouped Metrics
        metrics.update(pred, target)

        # Confusion Matrix
        cm.update(pred, target)

        # Closed ECE
        assert self.conf.classes.index("closed") == 1
        probs = torch.softmax(pred, dim=1)[:, 1]
        ece.update(probs, target)

        self.log_dict(
            {f"{train_test_val}/loss": loss},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch["pixels"]),
        )

        return loss

    def on_step_end(self, train_test_val: str) -> None:
        if train_test_val == "train":
            metrics = self.train_metrics
            cm = self.train_cm
            ece = self.train_ece
        elif train_test_val == "test":
            metrics = self.test_metrics
            cm = self.test_cm
            ece = self.val_ece
        else:
            metrics = self.val_metrics
            cm = self.val_cm
            ece = self.test_ece

        # compute all metrics
        results = metrics.compute()
        # extract and log per-class precision scalars
        if f"{train_test_val}/precision_perclass" in results:
            prec_vals = results.pop(f"{train_test_val}/precision_perclass")
            for idx, cls in enumerate(self.conf.classes):
                results[f"{train_test_val}/precision_{cls}"] = prec_vals[idx]
        # log the remaining metrics
        self.log_dict(results, sync_dist=True)
        metrics.reset()

        # 2) Compute & plot the confusion matrix
        fig, ax = plt.subplots(figsize=(4, 4))
        cm.plot(labels=list(self.conf.classes), ax=ax)
        # 3) Send the figure to TensorBoard
        # `self.logger` is a TensorBoardLogger
        tlog: TensorBoardLogger = self.logger  # type: ignore
        assert tlog.experiment is not None
        tlog.experiment.add_figure(f"cm/{train_test_val}", fig, global_step=self.current_epoch)
        plt.close(fig)
        cm.reset()

        # Log closed ECE
        results = ece.compute()
        self.log_dict({f"{train_test_val}/ece_closed": results}, sync_dist=True)
        ece.reset()

    def training_step(
        self, batch_target: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step("train", batch_target, batch_idx)

    def on_train_epoch_end(self) -> None:
        self.on_step_end("train")

    def validation_step(
        self,
        batch_target: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ):
        return self.step("val", batch_target, batch_idx)

    def on_validation_epoch_end(self):
        self.on_step_end("val")

    def test_step(
        self,
        batch_target: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ):
        return self.step("test", batch_target, batch_idx)

    def on_test_epoch_end(self):
        self.on_step_end("test")

    # ------------------------------------------------------------------
    #  Optimiser + LR schedule
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        # --------------------------------------------------------------
        # Compute LR scaling
        # --------------------------------------------------------------
        global_batch_size = self.conf.batch_size * self.conf.world_size * self.conf.grad_accum_steps
        batch_ratio = (global_batch_size / self.conf.base_lr_batch_size) ** 0.5
        lr = self.conf.lr * batch_ratio
        init_lr = self.conf.init_lr * batch_ratio
        min_lr = self.conf.min_lr * batch_ratio
        warmup_epochs = self.conf.warmup_epochs

        # --------------------------------------------------------------
        # Parameter groups: apply weight‑decay only where it matters
        # --------------------------------------------------------------
        decay, no_decay = [], []
        norm_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        )

        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                # biases OR parameters in a normalisation layer ⇒ no weight‑decay
                if name.endswith("bias") or isinstance(module, norm_modules):
                    no_decay.append(param)
                else:
                    decay.append(param)

        param_groups = [
            {"params": decay, "weight_decay": self.conf.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # --------------------------------------------------------------
        # Optimiser
        # --------------------------------------------------------------
        if self.conf.optimizer == "adamw":
            optimizer = optim.AdamW(param_groups, lr=lr)
        else:
            raise RuntimeError(f"Unexpected optimizer {self.conf.optimizer}")

        # --------------------------------------------------------------
        # Scheduler: Linear warm‑up  ➜  Cosine annealing
        # --------------------------------------------------------------
        schedulers, milestones = [], []
        if warmup_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=init_lr / lr,
                total_iters=warmup_epochs,
            )
            schedulers.append(warmup)
            milestones.append(warmup_epochs)

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.conf.epochs - warmup_epochs if warmup_epochs else self.conf.epochs,
            eta_min=min_lr,
        )
        schedulers.append(cosine)

        scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
