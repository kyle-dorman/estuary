from dataclasses import dataclass


@dataclass
class AugmentConfig:
    scale: tuple[float, float] = (0.9, 1.0)
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotation_p: float = 0.1
    salt_pepper_amount: tuple[float, float] = (0.01, 0.06)
    erasing_scale: tuple[float, float] = (0.02, 0.05)
    rain_number_of_drops: tuple[int, int] = (300, 500)
    rain_drop_height: tuple[int, int] = (5, 15)
    rain_drop_width: tuple[int, int] = (-5, 5)
    shade_intensity: tuple[float, float] = (-0.2, -0.05)
    shade_quantity: tuple[float, float] = (0.2, 0.4)
    fog_roughness: tuple[float, float] = (0.4, 0.6)
    fog_intensity: tuple[float, float] = (0.8, 0.9)
    illumination_gain: tuple[float, float] = (0.05, 0.2)
    posterize_bits: int = 5
    channel_shift_limit: float = 0.15
    plasma_brightness: tuple[float, float] = (0.1, 0.15)
    contrast: float = 0.3
    brightness: float = 0.1
    sharpness: float = 2.0
    gauss_std: float = 0.02
    blur_kernel_size: int = 7
    blur_sigma: tuple[float, float] = (0.5, 2.0)
    median_blur_kernel_size: int = 5
    box_blur_kernel_size: int = 5
