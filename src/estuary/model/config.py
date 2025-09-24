from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from estuary.constants import EIGHT_TO_4, FALSE_COLOR_4, FALSE_COLOR_8, RGB_4, RGB_8

CLASSES = ("closed", "open")


class Bands(Enum):
    RGB = "rgb"
    FALSE_COLOR = "false_color"
    EIGHT = "8"
    FOUR = "4"

    def eight_band_idxes(self) -> tuple[int, ...]:
        if self == Bands.FALSE_COLOR:
            return FALSE_COLOR_8
        elif self == Bands.RGB:
            return RGB_8
        elif self == Bands.EIGHT:
            return tuple(range(8))
        elif self == Bands.FOUR:
            return (7, 5, 3, 1)
        else:
            raise RuntimeError(f"Unexpected band type {self}")

    def num_channels(self):
        if self in [Bands.FALSE_COLOR, Bands.RGB]:
            return 3
        elif self == Bands.EIGHT:
            return 8
        elif self == Bands.FOUR:
            return 4
        else:
            raise RuntimeError(f"Unexpected band type {self}")

    def band_order(self, inpt_channels: int) -> tuple[int, ...]:
        if self == Bands.FALSE_COLOR:
            if inpt_channels == 4:
                return FALSE_COLOR_4
            else:
                return FALSE_COLOR_8

        elif self == Bands.RGB:
            if inpt_channels == 4:
                return RGB_4
            else:
                return RGB_8

        elif self == Bands.FOUR:
            if inpt_channels == 4:
                return tuple(range(4))
            else:
                return EIGHT_TO_4

        elif self == Bands.EIGHT:
            if inpt_channels == 4:
                return EIGHT_TO_4
            else:
                return tuple(range(8))
        else:
            raise RuntimeError(f"Unexpected band type {self}")


@dataclass
class EstuaryConfig:
    project: str = "estuary"
    seed: int = 42
    data: Path = Path("/UPDATEME")
    region_splits: Path = Path("/UPDATEME")
    model_training_root: Path = Path("/Users/kyledorman/data/results")

    classes: tuple[str, ...] = CLASSES
    bands: Bands = Bands.RGB
    devices: tuple[str, ...] = ("auto",)
    accelerator: str = "auto"
    compile: bool = False
    # Use False b/c TorchMetrics allocates too much memory otherwise
    deterministic: bool = False
    debug: bool = False

    epochs: int = 1
    grad_accum_steps: int = 1
    log_every_n_steps: int = 16 * 2
    log_image_every_n_epochs: int = 10
    precision: str = "bf16-mixed"
    batch_size: int = 16
    workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = True

    model_type: Literal["clay", "timm"] = "timm"
    model_name: str = "resnet18"
    pretrained: bool = True
    clay_encoder_weights: Path = Path("/Users/kyledorman/data/models/clay/clay-v1.5.ckpt")
    decoder_name: str = "conv"
    freeze_encoder: bool = True
    encoder_dim: int = 1024
    decoder_dim: int = 192
    decoder_depth: int = 4  # 4 for conv and 3 for attn
    decoder_heads: int = 2
    decoder_dim_head: int = 48
    decoder_mlp_ratio: int = 2
    dropout: float = 0.1
    train_size: int = 256
    val_size: int = 256
    world_size: int = 1

    smooth_factor: float = 0.05
    monitor_metric: str = "val/f1"
    monitor_mode: str = "max"

    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotation_p: float = 0.5
    metadata_path: Path = Path("/Users/kyledorman/data/models/clay/metadata.yaml")
    # /Volumes/x10pro/estuary/dataset/normalization/stats.json
    normalization_path: Path | None = None
    contrast: float = 0.15
    contrast_p: float = 1.0
    brightness: float = 0.15
    brightness_p: float = 1.0
    scale: tuple[float, float] = (0.9, 1.0)
    sharpness: float = 2.0
    sharpness_p: float = 0.1
    erasing_scale: tuple[float, float] = (0.02, 0.05)
    erasing_p: float = 0.1
    gauss_mean: float = 0.0
    gauss_std: float = 0.01
    gauss_p: float = 0.1
    blur_kernel_size: int = 7
    blur_sigma: tuple[float, float] = (0.0, 1.0)
    blur_p: float = 0.1
    channel_shift_limit: float = 0.2
    channel_shift_p: float = 0.1

    min_rows_per_split: int = 50  # sanity check

    lr: float = 1e-3
    base_lr_batch_size: int = 16
    warmup_epochs: int = 2
    init_lr: float = 2.0e-4
    min_lr: float = 1e-5
    patience: int = 5
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    gamma: float = 0.9
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str | None = "norm"
    loss_fn: str = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # class_weights will be computed from training set and injected
    use_class_weights: bool = True
    class_weights: tuple[float, ...] | None = None
