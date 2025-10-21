import logging
from pathlib import Path

import numpy as np
import rasterio
from numpy.ma.core import MaskedArray
from PIL import Image, ImageDraw, ImageFont

from estuary.constants import FALSE_COLOR_4, FALSE_COLOR_8

local_logger = logging.getLogger(__name__)


def tif_paths(directory: Path) -> list[Path]:
    return sorted([pth for pth in directory.iterdir() if pth.suffix == ".tif"])


def contrast_stretch(
    image: np.ndarray | MaskedArray, p_low: int = 2, p_high: int = 98
) -> np.ndarray:
    """Perform contrast stretching using percentiles."""
    image = image.astype(np.float32)
    orig_shape = image.shape
    if len(orig_shape) == 2:
        image = image[None]
    for idx in range(image.shape[0]):
        channel = image[idx]
        if isinstance(channel, MaskedArray):
            v_min, v_max = np.percentile(channel.compressed(), (p_low, p_high))
        else:
            v_min, v_max = np.percentile(channel, (p_low, p_high))

        image[idx] = np.clip((channel - v_min) / (v_max - v_min), 0, 1)

    if len(orig_shape) == 2:
        image = image[0]

    return image


def masked_contrast_stretch(
    image: np.ndarray, mask: np.ndarray, p_low: int | None = 2, p_high: int = 98
) -> np.ndarray:
    """Perform contrast stretching using percentiles."""
    image = image.astype(np.float32)
    orig_shape = image.shape
    if len(orig_shape) == 2:
        image = image[None]
    for idx in range(image.shape[0]):
        channel = image[idx]
        pp_low = p_low if p_low is not None else 0
        v_min, v_max = np.percentile(channel[mask], (pp_low, p_high))

        # If no p_low use 0 (absolute)
        if p_low is None:
            v_min = 0

        image[idx] = np.clip((channel - v_min) / (v_max - v_min), 0, 1)

    if len(orig_shape) == 2:
        image = image[0]

    return image


def broad_band(data: np.ndarray, nodata: np.ndarray) -> np.ndarray:
    red_recipe = np.mean(data[4:8], axis=0)
    green_recipe = np.mean(data[2:4], axis=0)
    blue_recipe = data[1]

    rgb = np.dstack((red_recipe, green_recipe, blue_recipe))

    rgb = masked_contrast_stretch(rgb.transpose((2, 0, 1)), ~nodata, p_low=0, p_high=99).transpose(
        (1, 2, 0)
    )

    rgb[nodata] = 0.0

    k = 2.2
    rgb = np.tanh(k * rgb) / np.tanh(k)

    img = np.array(rgb * 255, dtype=np.uint8)

    return img


def normalized_image_3_channel(
    data: np.ndarray, nodata: np.ndarray, bands: tuple[int, int, int]
) -> np.ndarray:
    img = np.array([data[b] for b in bands])
    img = masked_contrast_stretch(img, ~nodata, p_low=1, p_high=99)

    for i in range(3):
        img[i][nodata] = 0

    return img


def false_color(data: np.ndarray, nodata: np.ndarray):
    channels = FALSE_COLOR_4 if len(data) == 4 else FALSE_COLOR_8
    img = normalized_image_3_channel(data, nodata, channels).transpose((1, 2, 0))

    k = 1.5
    img = np.tanh(k * img) / np.tanh(k)

    img = np.array(img * 255, dtype=np.uint8)

    return img


def tif_to_rgb(pth: Path) -> np.ndarray:
    with rasterio.open(pth) as src:
        data = src.read(out_dtype=np.float32)
        nodata = src.read_masks(1) == 0
        if nodata.all():
            return np.zeros((*nodata.shape, 3), dtype=np.uint8)

        return false_color(data, nodata)


def setup_logger(save_dir: Path | None = None, log_filename: str = "log.log"):
    # Remove base handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set level for logger
    root_logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if save_dir is not None:
        # Create handler
        file_handler = logging.FileHandler(save_dir / log_filename)  # Logs to a file

        # Attach formatter to the handler
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        root_logger.addHandler(file_handler)

    # Create handler
    console_handler = logging.StreamHandler()  # Logs to console

    # Attach formatter to the handler
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    root_logger.addHandler(console_handler)


def draw_border(img: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Draw a colored border."""
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Border matching class color
    draw.rectangle([0, 0, w - 1, h - 1], outline=color + (255,), width=4)

    return img


def draw_label(
    img: Image.Image, text: str, color: tuple[int, int, int], add_border=True
) -> Image.Image:
    """Draw a semi-transparent banner with outlined text, and optional colored border."""
    # Optional: try a nicer font; fall back to default if not available
    try:
        FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    except Exception:
        FONT = ImageFont.load_default()

    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Banner box
    pad_x, pad_y = 10, 8
    text_w, text_h = draw.textbbox((0, 0), text, font=FONT)[2:]
    box_w = min(w - 2 * pad_x, text_w + 2 * pad_x)
    box_h = text_h + 2 * pad_y

    # Top-left anchor for banner
    x0, y0 = pad_x, pad_y
    x1, y1 = x0 + box_w, y0 + box_h

    # Semi-transparent dark banner
    draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(0, 0, 0, 110))

    # Outlined text (stroke) for readability
    draw.text(
        (x0 + pad_x, y0 + pad_y),
        text,
        font=FONT,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 220),
    )

    # Optional border matching class color
    if add_border:
        draw.rectangle([0, 0, w - 1, h - 1], outline=color + (255,), width=4)

    return img
