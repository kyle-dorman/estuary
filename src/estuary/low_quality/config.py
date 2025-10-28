from dataclasses import dataclass
from pathlib import Path

from estuary.util.bands import Bands

CLASSES = ("bad", "good")


@dataclass
class QualityConfig:
    project: str = "estuary_quality"
    seed: int = 42
    data: Path = Path("/UPDATEME")
    model_training_root: Path = Path("/Users/kyledorman/data/results")

    classes: tuple[str, ...] = CLASSES
    bands: Bands = Bands.FALSE_COLOR
    devices: tuple[str, ...] = ("auto",)
    accelerator: str = "auto"
    compile: bool = False
    # Use False b/c TorchMetrics allocates too much memory otherwise
    deterministic: bool = False
    debug: bool = False

    epochs: int = 1
    grad_accum_steps: int = 1
    log_every_n_steps: int = 4
    precision: str = "16-mixed"
    batch_size: int = 16
    workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 0
    preview_n: int = 9
    preview_channels: tuple[int, int, int] = (0, 1, 2)

    model_name: str = "convnext_tiny.dinov3_lvd1689m"
    global_pool: str = "avg"
    pretrained: bool = True
    dropout: float = 0.15
    drop_path: float = 0.1
    train_size: int = 224
    val_size: int = 224
    world_size: int = 1

    monitor_metric: str = "val/f1"
    monitor_mode: str = "max"

    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotation_p: float = 0.1
    normalization_path: Path | None = None
    contrast: float = 0.05
    contrast_p: float = 0.05
    brightness: float = 0.05
    brightness_p: float = 0.05
    scale: tuple[float, float] = (0.9, 1.0)
    sharpness: float = 1.0
    sharpness_p: float = 0.05
    channel_shift_limit: float = 0.05
    channel_shift_p: float = 0.05

    lr: float = 5e-5
    base_lr_batch_size: int = 128
    warmup_epochs: int = 2
    init_lr_scale: float = 1e-1
    min_lr_scale: float = 5e-2
    backbone_lr_scale: float | None = None
    patience: int = 5
    optimizer: str = "adamw"
    weight_decay: float = 3e-4
    scheduler: str = "cosine"

    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str | None = "norm"
    loss_fn: str = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
