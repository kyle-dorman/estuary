import datetime
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from estuary.util.bands import Bands


def parse_dt_from_pth(pth: Path) -> datetime.datetime:
    """Parse acquisition datetime from file stem prefix YYYYMMDD_HHMMSS_*"""
    datetime_str = "_".join(pth.stem.split("_")[:2])
    date_format = "%Y%m%d_%H%M%S"
    return datetime.datetime.strptime(datetime_str, date_format)


def cpu_count() -> int:
    cnt = os.cpu_count()
    if cnt is None:
        return 0
    return cnt


def normalize_latlon(lat: float, lon: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Encode latitude/longitude onto the unit circle for model input."""
    lat_r = lat * math.pi / 180
    lon_r = lon * math.pi / 180
    return (math.sin(lat_r), math.cos(lat_r)), (math.sin(lon_r), math.cos(lon_r))


def normalize_timestamp(date: datetime.datetime) -> tuple[tuple[float, float], tuple[float, float]]:
    """Encode ISO week-of-year and hour-of-day as sin/cos pairs."""
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = date.hour * 2 * math.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


@dataclass
class NormalizationStats:
    power_scale: bool
    max_pixel_value: int
    mean: np.ndarray
    std: np.ndarray
    lambdas: np.ndarray


def load_normalization(normalization_path: Path, bands: Bands) -> NormalizationStats:
    with open(normalization_path) as f:
        stats = json.load(f)
        power_scale = bool(stats["power_scale"])
        max_pixel_value = int(stats["max_raw_pixel_value"])

        idxes = bands.eight_band_idxes()
        mean = np.array([stats["means"][i] for i in idxes])
        std = np.array([stats["stds"][i] for i in idxes])
        lambdas = np.array([stats["lambdas"][i] for i in idxes])

        return NormalizationStats(
            power_scale=power_scale,
            max_pixel_value=max_pixel_value,
            mean=mean,
            std=std,
            lambdas=lambdas,
        )
