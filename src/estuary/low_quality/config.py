from dataclasses import dataclass, field
from pathlib import Path

from estuary.util.bands import Bands
from estuary.util.config import AugmentConfig

CLASSES = ("good", "bad")


@dataclass
class QualityConfig:
    project: str = "estuary_quality"
    seed: int = 42
    data: Path = Path("/UPDATEME")
    val_data: Path | None = None
    test_data: Path = Path("/UPDATEME")
    # How to split train/val/test. One of ["dataset", "yearly"]
    split_method: str = "yearly"
    val_year: int | None = None
    model_training_root: Path = Path("/Users/kyledorman/data/results")
    normalization_path: Path | None = None
    checkpoint_path: Path | None = None

    classes: tuple[str, ...] = CLASSES
    bands: Bands = Bands.FALSE_COLOR
    devices: tuple[str, ...] = ("auto",)
    accelerator: str = "auto"
    compile: bool = False
    # Use False b/c TorchMetrics allocates too much memory otherwise
    deterministic: bool = False
    debug: bool = False
    pct_low_quality: float = 0.5

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
    freeze_encoder: bool = False
    encoder_checkpoint_path: Path | None = None
    train_size: int = 224
    val_size: int = 224
    world_size: int = 1

    monitor_metric: str = "val/f1"
    monitor_mode: str = "max"

    # Augmentation Params
    aug: AugmentConfig = field(default_factory=AugmentConfig)

    # Optimizer Params
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

    # Loss Params
    loss_fn: str = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # class_weights will be computed from training set and injected
    use_class_weights: bool = False
    class_weights: tuple[float, ...] | None = None
