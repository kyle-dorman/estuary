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
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.ops import sigmoid_focal_loss

from estuary.clay.classifier import Classifier, ConvDecoder, TransformerDecoder
from estuary.clay.config import EstuaryConfig

logger = logging.getLogger(__name__)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float, gamma: float, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return sigmoid_focal_loss(
            inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )


class EstuaryModule(LightningModule):
    def __init__(self, conf: EstuaryConfig):
        self.conf = conf
        self.num_classes = len(conf.classes)
        self.is_binary = self.num_classes == 2

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

        if self.is_binary:
            # Binary metrics operate on probabilities in [0,1]
            metrics_dict = {
                "f1": BinaryF1Score(),
                "accuracy": BinaryAccuracy(),
                "auroc": BinaryAUROC(),
            }
            self.train_cm = BinaryConfusionMatrix()
            self.val_cm = BinaryConfusionMatrix()
            self.test_cm = BinaryConfusionMatrix()
            self.train_ece = BinaryCalibrationError()
            self.val_ece = BinaryCalibrationError()
            self.test_ece = BinaryCalibrationError()
        else:
            metrics_dict = {
                "f1": MulticlassF1Score(num_classes=self.num_classes),
                "precision": MulticlassPrecision(num_classes=self.num_classes),
                "recall": MulticlassRecall(num_classes=self.num_classes),
                "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
                "cohenkappa": MulticlassCohenKappa(num_classes=self.num_classes),
                "auroc": MulticlassAUROC(num_classes=self.num_classes),
            }
            # per-class precision vector (multiclass only)
            metrics_dict["precision_perclass"] = MulticlassPrecision(
                num_classes=self.num_classes, average="none"
            )
            self.train_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
            self.val_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
            self.test_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
            self.train_ece = BinaryCalibrationError()
            self.val_ece = BinaryCalibrationError()
            self.test_ece = BinaryCalibrationError()

        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix="train/").to(self.device)
        self.val_metrics = metrics.clone(prefix="val/").to(self.device)
        self.test_metrics = metrics.clone(prefix="test/").to(self.device)

        # Loss
        weights = (
            torch.tensor(conf.class_weights, device=self.device)
            if conf.class_weights is not None
            else None
        )
        if self.is_binary:
            if conf.loss_fn == "ce":
                # Single-logit BCE with logits. Turn class weights into pos_weight if provided.
                pos_weight = None
                if weights is not None and len(weights) == 2:
                    # CE weights ~ [w0, w1]; BCE pos_weight scales positives.
                    w0, w1 = weights[0].item(), weights[1].item()
                    if w0 > 0:
                        pos_weight = torch.tensor([w1 / w0], device=self.device)
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            elif conf.loss_fn == "focal":
                self.loss_fn = FocalLoss(gamma=conf.focal_gamma, alpha=conf.focal_alpha)
            else:
                raise RuntimeError(f"Invalid loss_fn {conf.loss_fn}")
        else:
            if conf.loss_fn == "ce":
                self.loss_fn = nn.CrossEntropyLoss(
                    label_smoothing=conf.smooth_factor, weight=weights
                )
            elif conf.loss_fn == "focal":
                self.loss_fn = FocalLoss(gamma=conf.focal_gamma, alpha=conf.focal_alpha)
            else:
                raise RuntimeError(f"Invalid loss_fn {conf.loss_fn}")

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
        logits: torch.Tensor = self(batch)

        if self.is_binary:
            # logits: [B, 1]; targets: int {0,1}
            logits = logits.view(-1)
            target_f = target.float()
            loss = self.loss_fn(logits, target_f)
            probs_pos = torch.sigmoid(logits)

            # Metrics expect probabilities for binary tasks
            metrics.update(probs_pos, target)
            # Confusion matrix uses hard labels
            preds_label = (probs_pos >= 0.5).long()
            cm.update(preds_label, target)

            # ECE on positive class
            ece.update(probs_pos, target)
        else:
            # Multiclass: logits [B, C]
            loss = self.loss_fn(logits, target)
            metrics.update(logits, target)  # torchmetrics multiclass can accept logits
            cm.update(logits.argmax(dim=1), target)

            # Closed ECE assumes index 1 is 'closed'
            assert self.conf.classes.index("closed") == 1
            probs = torch.softmax(logits, dim=1)[:, 1]
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
