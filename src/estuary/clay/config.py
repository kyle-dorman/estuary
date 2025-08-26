from dataclasses import dataclass
from pathlib import Path

CLASSES = ("open", "closed")


@dataclass
class EstuaryConfig:
    project: str = "estuary"
    seed: int = 42
    data: Path = Path("/UPDATEME")
    region_crops_json: Path = Path("/UPDATEME")
    model_training_root: Path = Path("/Users/kyledorman/data/results")
    classes: tuple[str, ...] = CLASSES
    bands: int = 4
    devices: tuple[str, ...] = ("auto",)
    accelerator: str = "auto"
    # Use False b/c TorchMetrics allocates too much memory otherwise
    deterministic: bool = False
    debug: bool = False

    epochs: int = 1
    chip_size: int = 256
    world_size: int = 1
    grad_accum_steps: int = 1
    log_every_n_steps: int = 16 * 2
    log_image_every_n_epochs: int = 10
    precision: str = "bf16-mixed"
    batch_size: int = 16
    workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = True

    encoder_weights: Path = Path("/Users/kyledorman/data/models/clay/clay-v1.5.ckpt")
    decoder_name: str = "conv"
    freeze_encoder: bool = True
    encoder_dim: int = 1024
    decoder_dim: int = 192
    decoder_depth: int = 4  # 4 for conv and 3 for attn
    decoder_heads: int = 2
    decoder_dim_head: int = 48
    decoder_mlp_ratio: int = 2
    dropout: float = 0.1

    smooth_factor: float = 0.05
    monitor_metric: str = "val/f1"
    monitor_mode: str = "max"

    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    metadata_path: Path = Path("/Users/kyledorman/data/models/clay/metadata.yaml")

    test_year: int | None = None  # default: max year in data
    val_year: int | None = None  # default: second-max year in data
    holdout_region: str | None = None
    require_all_classes: bool = True  # sanity check per split
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

    # class_weights will be computed from training set and injected
    class_weights: tuple[float, ...] | None = None
