import logging
from pathlib import Path

import numpy as np
import rasterio
from numpy.ma.core import MaskedArray

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


def tif_to_rgb(pth: Path) -> np.ndarray:
    with rasterio.open(pth) as src:
        data = src.read(out_dtype=np.float32)
        nodata = src.read(1, masked=True).mask
    data = np.log10(data + 1)
    imgd = masked_contrast_stretch(data, ~nodata, p_low=1, p_high=99)

    img = np.zeros((*imgd.shape[1:], 3), dtype=imgd.dtype)
    img[:, :, 0] = imgd[3]
    img[:, :, 1] = imgd[2]
    img[:, :, 2] = imgd[:2].max(axis=0)
    img[nodata] = 0
    img = np.array(img * 255, dtype=np.uint8)
    return img


def setup_logger(
    logger: logging.Logger, save_dir: Path | None = None, log_filename: str = "log.log"
):
    # Remove base handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set level for logger
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if save_dir is not None:
        # Create handler
        file_handler = logging.FileHandler(save_dir / log_filename)  # Logs to a file

        # Attach formatter to the handler
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)

    # Create handler
    console_handler = logging.StreamHandler()  # Logs to console

    # Attach formatter to the handler
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
