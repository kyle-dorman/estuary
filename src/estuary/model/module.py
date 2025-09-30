import logging

import kornia.augmentation as K
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
from torchvision.utils import make_grid

from estuary.clay.classifier import ClayClassifier, ClayConvDecoder, ClayTransformerDecoder
from estuary.model.config import EstuaryConfig, ModelType
from estuary.model.data import load_normalization
from estuary.model.timm_model import TimmModel
from estuary.model.transforms import contrast_stretch_torch

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
        self.is_binary = len(conf.classes) == 2
        if self.is_binary:
            self.num_classes = 1
        else:
            self.num_classes = len(conf.classes)
        self.norm_stats = load_normalization(conf)

        super().__init__()
        self.save_hyperparameters(conf)

        if conf.model_type == ModelType.CLAY:
            clay_model = ClayMAEModule.load_from_checkpoint(
                conf.clay_encoder_weights,
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
            decoder = (
                ClayConvDecoder(conf, self.num_classes)
                if conf.decoder_name == "conv"
                else ClayTransformerDecoder(conf, self.num_classes)
            )
            clf = ClayClassifier(encoder, decoder)
        elif conf.model_type == ModelType.TIMM:
            clf = TimmModel(conf, self.num_classes)
        else:
            raise RuntimeError(f"Unsupported model_type {conf.model_type}")

        if conf.debug or not conf.compile:
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
        train_test_val: str,
        batch: dict[str, torch.Tensor],
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

        logits: torch.Tensor = self(batch)
        target = batch["label"]

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

            # Open ECE assumes index 1 is 'open'
            assert self.conf.classes.index("open") == 1
            probs = torch.softmax(logits, dim=1)[:, 1]
            ece.update(probs, target)

        self.log_dict(
            {f"{train_test_val}/loss": loss},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch["image"]),
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

        # Log open ECE
        results = ece.compute()
        self.log_dict({f"{train_test_val}/ece_open": results}, sync_dist=True)
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
        # Parameter groups: apply weight‑decay only where it matters
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
        # Scheduler: Linear warm‑up  ➜  Cosine annealing
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

        cosine_epochs = self.conf.epochs - warmup_epochs - self.conf.flat_epochs
        assert cosine_epochs > 0
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=lr * self.conf.min_lr_scale,
        )
        schedulers.append(cosine)

        if self.conf.flat_epochs > 0:
            flat = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=self.conf.min_lr_scale, total_iters=self.conf.flat_epochs
            )
            schedulers.append(flat)
            milestones.append(warmup_epochs + cosine_epochs)

        if len(schedulers) > 1:
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
        else:
            scheduler = schedulers[0]

        return [optimizer], [scheduler]
