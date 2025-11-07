import logging

import kornia.augmentation as K
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from timm.layers import trunc_normal_
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCalibrationError,
    BinaryConfusionMatrix,
    BinaryF1Score,
)
from torchvision.utils import make_grid

from estuary.low_quality.config import QualityConfig
from estuary.low_quality.timm_model import TimmModel
from estuary.model.module import EstuaryModule
from estuary.util.data import load_normalization
from estuary.util.nn import FocalLoss
from estuary.util.transforms import contrast_stretch_torch

logger = logging.getLogger(__name__)


def _init_weights(module: nn.Module, head_init_scale: float = 1.0) -> None:
    """Initialize model weights.

    Args:
        module: Module to initialize.
        head_init_scale: Scale factor for head initialization.
    """
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
        module.weight.data.mul_(head_init_scale)
        module.bias.data.mul_(head_init_scale)


def load_encorder_from_path(conf: QualityConfig, num_classes: int) -> nn.Module:
    assert conf.encoder_checkpoint_path is not None

    module = EstuaryModule.load_from_checkpoint(
        conf.encoder_checkpoint_path, compile=False, drop_path=conf.drop_path, dropout=conf.dropout
    )
    model: TimmModel = module.model  # type: ignore

    model.model.reset_classifier(num_classes, conf.global_pool)  # type: ignore
    for child_name, child_module in model.model.head.named_children():  # type: ignore
        logger.info(f"Setting weights for {child_name}")
        _init_weights(child_module)

    if conf.freeze_encoder:
        head_names = ["head", "classifier"]

        if conf.freeze_encoder:
            for name, param in model.named_parameters():
                if not any(k in name for k in head_names):
                    param.requires_grad = False

    return model  # type: ignore


class LowQualityModule(LightningModule):
    def __init__(self, conf: QualityConfig):
        self.conf = conf
        assert len(self.conf.classes) == 2
        self.num_classes = 1
        assert conf.normalization_path is not None
        self.norm_stats = load_normalization(conf.normalization_path, conf.bands)

        super().__init__()
        self.save_hyperparameters(conf)

        if conf.encoder_checkpoint_path is not None:
            clf = load_encorder_from_path(conf, self.num_classes)
        else:
            clf = TimmModel(conf, self.num_classes)

        if conf.debug or not conf.compile:
            self.model = clf
        else:
            self.model = torch.compile(clf)  # type: ignore

        logger.info("Creating metrics")

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

        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix="train/").to(self.device)
        self.val_metrics = metrics.clone(prefix="val/").to(self.device)
        self.test_metrics = metrics.clone(prefix="test/").to(self.device)

        if conf.loss_fn == "ce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif conf.loss_fn == "focal":
            self.loss_fn = FocalLoss(gamma=conf.focal_gamma, alpha=conf.focal_alpha)
        else:
            raise RuntimeError(f"Invalid loss_fn {conf.loss_fn}")

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(x)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return K.Denormalize(mean=self.norm_stats.mean.tolist(), std=self.norm_stats.std.tolist())(
            x
        )

    def _log_batch_images(self, imgs: torch.Tensor, split: str, step: int) -> None:
        """Log the first N images from the first batch to TensorBoard.
        Expects imgs shape (B, C, H, W). Uses first 3 channels by default.
        """
        try:
            if not isinstance(self.logger, TensorBoardLogger):
                return
            if imgs.ndim != 4:
                return
            # N and channels to preview (fall back to sensible defaults)
            n_show: int = self.conf.preview_n
            ch_idx = self.conf.preview_channels

            imgs = self.denormalize(imgs.detach())
            B, C, H, W = imgs.shape
            n = min(n_show, B)

            # Select channels (handle multispectral gracefully)
            sel = list(ch_idx) if len(ch_idx) == 3 and max(ch_idx) < C else [0, 1, 2]
            imgs_sel = imgs[:n, sel, :, :]

            # Move to CPU for logging, normalize, clamp to [0,1]
            imgs_vis = contrast_stretch_torch(imgs_sel.to("cpu"))

            grid = make_grid(imgs_vis, nrow=min(n, 3), padding=2)
            tlog: TensorBoardLogger = self.logger  # type: ignore
            assert tlog.experiment is not None
            tlog.experiment.add_image(f"preview/{split}", grid, global_step=step)
        except Exception as e:
            logger.warning(f"Failed to log preview images: {e}")

    def step(
        self,
        train_val: str,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        if train_val == "train":
            metrics = self.train_metrics
            cm = self.train_cm
            ece = self.train_ece
        elif train_val == "test":
            metrics = self.test_metrics
            cm = self.test_cm
            ece = self.test_ece
        else:
            metrics = self.val_metrics
            cm = self.val_cm
            ece = self.val_ece

        logits: torch.Tensor = self(batch)
        target = batch["label"]

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

        self.log_dict(
            {f"{train_val}/loss": loss},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch["image"]),
        )

        return loss

    def on_step_end(self, train_val: str) -> None:
        if train_val == "train":
            metrics = self.train_metrics
            cm = self.train_cm
            ece = self.train_ece
        elif train_val == "test":
            metrics = self.test_metrics
            cm = self.test_cm
            ece = self.test_ece
        else:
            metrics = self.val_metrics
            cm = self.val_cm
            ece = self.val_ece

        # compute all metrics
        results = metrics.compute()
        # extract and log per-class precision scalars
        if f"{train_val}/precision_perclass" in results:
            prec_vals = results.pop(f"{train_val}/precision_perclass")
            for idx, cls in enumerate(self.conf.classes):
                results[f"{train_val}/precision_{cls}"] = prec_vals[idx]
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
        tlog.experiment.add_figure(f"cm/{train_val}", fig, global_step=self.current_epoch)
        plt.close(fig)
        cm.reset()

        # Log open ECE
        results = ece.compute()
        self.log_dict({f"{train_val}/ece_unsure": results}, sync_dist=True)
        ece.reset()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if batch_idx == 0:
            self._log_batch_images(batch["image"], split="train", step=self.current_epoch)
        return self.step("train", batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        self.on_step_end("train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ):
        if batch_idx == 0:
            self._log_batch_images(batch["image"], split="val", step=self.current_epoch)
        return self.step("val", batch, batch_idx)

    def on_validation_epoch_end(self):
        self.on_step_end("val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ):
        if batch_idx == 0:
            self._log_batch_images(batch["image"], split="test", step=self.current_epoch)
        return self.step("test", batch, batch_idx)

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
        backbone_lr = (
            self.conf.backbone_lr_scale * lr if self.conf.backbone_lr_scale is not None else None
        )

        # --------------------------------------------------------------
        # Parameter groups: apply weight-decay only where it matters
        # --------------------------------------------------------------
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

        # Helper: is this a normalization/bias param?
        def is_norm_or_bias(module, name):
            return name.endswith("bias") or isinstance(module, norm_modules)

        backbone_decay, backbone_no_decay, head_decay, head_no_decay = [], [], [], []
        # Traverse all named parameters with their modules
        for module_name, module in self.model.named_modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                param_full_name = f"{module_name}.{name}" if module_name else name
                is_head = ("head" in param_full_name) or ("classifier" in param_full_name)
                if is_norm_or_bias(module, name):
                    if is_head:
                        head_no_decay.append(param)
                    else:
                        backbone_no_decay.append(param)
                else:
                    if is_head:
                        head_decay.append(param)
                    else:
                        backbone_decay.append(param)

        # Build param groups: backbone and head, each with decay/no_decay, each with their own LR
        param_groups = []
        # backbone group
        if backbone_decay or backbone_no_decay:
            param_groups.append(
                {
                    "params": backbone_decay,
                    "weight_decay": self.conf.weight_decay,
                    "lr": backbone_lr if backbone_lr is not None else lr,
                }
            )
            param_groups.append(
                {
                    "params": backbone_no_decay,
                    "weight_decay": 0.0,
                    "lr": backbone_lr if backbone_lr is not None else lr,
                }
            )
        # head group
        if head_decay or head_no_decay:
            param_groups.append(
                {
                    "params": head_decay,
                    "weight_decay": self.conf.weight_decay,
                    "lr": lr,
                }
            )
            param_groups.append(
                {
                    "params": head_no_decay,
                    "weight_decay": 0.0,
                    "lr": lr,
                }
            )

        # Remove empty param groups
        param_groups = [g for g in param_groups if g["params"]]

        # --------------------------------------------------------------
        # Optimiser
        # --------------------------------------------------------------
        if self.conf.optimizer == "adamw":
            optimizer = optim.AdamW(param_groups)
        else:
            raise RuntimeError(f"Unexpected optimizer {self.conf.optimizer}")

        # --------------------------------------------------------------
        # Scheduler: Linear warm-up  ➜  Cosine annealing
        # --------------------------------------------------------------
        schedulers, milestones = [], []
        warmup_epochs = self.conf.warmup_epochs
        if self.conf.epochs - warmup_epochs <= 0:
            warmup_epochs = 0
        if warmup_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.conf.init_lr_scale,
                total_iters=warmup_epochs,
            )
            schedulers.append(warmup)
            milestones.append(warmup_epochs)

        if self.conf.scheduler == "cosine":
            cosine_epochs = self.conf.epochs - warmup_epochs
            assert cosine_epochs > 0
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=lr * self.conf.min_lr_scale,
            )
            schedulers.append(cosine)
        else:
            raise RuntimeError(f"Unsuported scheduler {self.conf.scheduler}")

        if len(schedulers) > 1:
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
        else:
            scheduler = schedulers[0]

        return [optimizer], [scheduler]
