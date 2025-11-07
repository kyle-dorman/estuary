from dataclasses import dataclass
from pathlib import Path

from estuary.util.bands import Bands

CLASSES = ("good", "bad")


@dataclass
class QualityConfig:
    project: str = "estuary_quality"
    seed: int = 42
    data: Path = Path("/UPDATEME")
    test_data: Path = Path("/UPDATEME")
    val_data: Path = Path("/UPDATEME")
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
    freeze_encoder: bool = False
    encoder_checkpoint_path: Path | None = None
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

    # Misalignment parameters
    misaligned_angle_min_deg: int = 0
    misaligned_angle_max_deg: int = 45
    misaligned_edge_shift_min_px: float = 10.0
    misaligned_edge_shift_max_px: float = 30.0
    misaligned_offset_min_px: float = 5.0
    misaligned_offset_max_px: float = 10.0
    misaligned_sign_flip_prob: float = 0.5
    misaligned_crop_border_px: int = 20

    # Noised Image Params
    read_noise_std_dev_min: float = 0.1
    read_noise_std_dev_max: float = 0.3
    noise_scale_min: int = 10
    noise_scale_max: int = 100

    # Hazy Image Params
    hazy_kernel_scale: int = 40
    hazy_read_noise_std_dev_min: float = 0.01
    hazy_read_noise_std_dev_max: float = 0.02
    hazy_noise_scale_min: int = 5000
    hazy_noise_scale_max: int = 10000
    hazy_beta_min: float = 5.0
    hazy_beta_max: float = 7.0
    hazy_strength_min: float = 0.9
    hazy_strength_max: float = 1.0
    hazy_posterize: int = 5

    # Cloudy Image Params
    cloudy_zero_prob: float = 0.01
    cloudy_kernel_scale: int = 40
    cloudy_sun_shift_min: int = -20
    cloudy_sun_shift_max: int = 20
    cloudy_coverage_min: float = 0.9
    cloudy_coverage_max: float = 1.0
    cloudy_softness_min: float = 0.2
    cloudy_softness_max: float = 0.8
    cloudy_brightness_min: float = 0.9
    cloudy_brightness_max: float = 1.0
    cloudy_shadow_min: float = 0.1
    cloudy_shadow_max: float = 0.4
    cloudy_transparency_lo_min: float = 0.7
    cloudy_transparency_lo_max: float = 0.8
    # cloudy_hole_soft_min: float = 1.0
    # cloudy_hole_soft_max: float = 9.0
    cloudy_alpha_min: float = 0.9
    cloudy_alpha_max: float = 1.0
    cloudy_edge_pow_min: float = 0.9
    cloudy_edge_pow_max: float = 1.5
    cloudy_tint_min: float = 0.95
    cloudy_tint_max: float = 1.0
    cloudy_add_noise_prob: float = 0.2
    cloudy_read_noise_std_min: float = 0.01
    cloudy_read_noise_std_max: float = 0.02
    cloudy_noise_scale_min: int = 1000
    cloudy_noise_scale_max: int = 5000

    # Fogged Image Params
    fogged_chroma_dx: tuple[int, int, int] = (0, 1, 0)
    fogged_chroma_dy: tuple[int, int, int] = (0, 0, -1)
    fogged_apply_haze: bool = True
    fogged_haze_beta: float = 3.0
    fogged_haze_A: tuple[float, float, float] = (0.9, 0.95, 1.0)
    fogged_haze_strength: float = 0.8
    fogged_apply_defocus: bool = True
    fogged_defocus_radius: int = 4
    fogged_defocus_down_factor: int = 4
    fogged_apply_speckle_noise_chance: float = 0.5
    fogged_contrast_chance: float = 0.8
    fogged_contrast_mul: float = 0.8
    fogged_quantize_8bit_chance: float = 0.5

    # Directional Sunlight Params
    sun_dir_min_deg: int = -45
    sun_dir_max_deg: int = 45
    sun_intensity_min: float = 0.45  # stronger base ramp
    sun_intensity_max: float = 0.70
    sun_veil_min: float = 0.20  # more veiling glare
    sun_veil_max: float = 0.35
    sun_warm_tint: tuple[float, float, float] = (1.0, 0.975, 0.93)
    sun_ramp_gamma: float = 0.15  # smaller gamma -> brighter sun side
    sun_white_push: float = 1.1  # moderate clipping toward white
    sun_veil_gain: float = 2.8  # stronger nonlinear haze falloff
    sun_exposure_gain: float = 0.2  # slight lift overall in lit region
    sun_bloom_quantile: float = 0.87  # bloom engages on more highlights
    sun_bloom_sigma: float = 11.0  # larger blur for wide bloom
    sun_bloom_strength: float = 0.50  # add-back amount
