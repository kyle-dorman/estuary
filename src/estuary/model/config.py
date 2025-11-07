from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from estuary.util.bands import Bands

CLASSES = ("closed", "open")


# ModelType Enum for selecting model type
class ModelType(Enum):
    CLAY = "clay"
    TIMM = "timm"


@dataclass
class EstuaryConfig:
    project: str = "estuary"
    seed: int = 42
    data: Path = Path("/UPDATEME")
    model_training_root: Path = Path("/Users/kyledorman/data/results")
    # How to split train/val/test. One of ["region", "crossval", "yearly"]
    split_method: str = "region"
    region_splits: Path = Path("/UPDATEME")
    cv_folds: int = 0
    cv_index: int = 0
    val_year: int | None = None
    test_year: int | None = None

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

    model_type: ModelType = ModelType.TIMM
    model_name: str = "convnext_tiny.dinov3_lvd1689m"
    pretrained: bool = True
    clay_encoder_weights: Path = Path("/Users/kyledorman/data/models/clay/clay-v1.5.ckpt")
    metadata_path: Path = Path("/Users/kyledorman/data/models/clay/metadata.yaml")
    decoder_name: str = "conv"
    freeze_encoder: bool = True
    encoder_dim: int = 1024
    decoder_dim: int = 192
    decoder_depth: int = 4  # 4 for conv and 3 for attn
    decoder_heads: int = 2
    decoder_dim_head: int = 48
    decoder_mlp_ratio: int = 2
    global_pool: str = "avg"
    lse_beta: float = 10.0
    dropout: float = 0.15
    drop_path: float = 0.1
    train_size: int = 224
    val_size: int = 224
    world_size: int = 1

    smooth_factor: float = 0.0
    perch_smooth_factor: float = 0.0
    monitor_metric: str = "val/f1"
    monitor_mode: str = "max"

    normalization_path: Path | None = None
    scale: tuple[float, float] = (0.9, 1.0)
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotation_p: float = 0.1
    salt_pepper_amount: tuple[float, float] = (0.01, 0.06)
    erasing_scale: tuple[float, float] = (0.02, 0.05)
    rain_number_of_drops: tuple[int, int] = (300, 700)
    rain_drop_height: tuple[int, int] = (5, 20)
    rain_drop_width: tuple[int, int] = (-5, 5)
    shade_intensity: tuple[float, float] = (-0.5, -0.1)
    shade_quantity: tuple[float, float] = (0.2, 0.5)
    fog_roughness: tuple[float, float] = (0.4, 0.6)
    fog_intensity: tuple[float, float] = (1.0, 1.0)
    illumination_gain: tuple[float, float] = (0.1, 0.5)
    posterize_bits: int = 5
    channel_shift_limit: float = 0.3
    plasma_brightness: tuple[float, float] = (0.1, 0.15)
    contrast: float = 0.5
    brightness: float = 0.2
    sharpness: float = 1.0
    gauss_std: float = 0.05
    blur_kernel_size: int = 7
    blur_sigma: tuple[float, float] = (0.5, 2.0)
    median_blur_kernel_size: int = 5
    box_blur_kernel_size: int = 5

    lr: float = 5e-5
    base_lr_batch_size: int = 128
    warmup_epochs: int = 2
    flat_epochs: int = 0
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

    # class_weights will be computed from training set and injected
    use_class_weights: bool = False
    class_weights: tuple[float, ...] | None = None
