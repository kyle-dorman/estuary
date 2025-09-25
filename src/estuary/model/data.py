import datetime
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kornia.augmentation as K
import numpy as np
import pandas as pd
import rasterio
import torch
from kornia.constants import Resample
from lightning.pytorch import LightningDataModule
from pyproj import Transformer
from torch.utils.data import DataLoader, Dataset

from estuary.model.config import (
    Bands,
    EstuaryConfig,
)
from estuary.model.transforms import PowerTransformTorch, RandomChannelShift, ScaleNormalization

logger = logging.getLogger(__name__)


def num_workers(conf: EstuaryConfig) -> int:
    # number of CUDA devices
    nd = max(1, conf.world_size)
    per_gpu_count = cpu_count() // nd

    if conf.workers == -1:
        return per_gpu_count
    # number of workers
    nw = min([per_gpu_count, conf.workers])

    return nw


def cpu_count() -> int:
    cnt = os.cpu_count()
    if cnt is None:
        return 0
    return cnt


def load_labels(conf: EstuaryConfig) -> pd.DataFrame:
    df = pd.read_csv(conf.data)

    if "perched open" not in conf.classes:
        df.loc[df.label == "perched open", "label"] = "open"

    cls_diff = set(df.label.unique()) - set(conf.classes)
    if cls_diff:
        logger.warning(f"Some label classes will be ignored {cls_diff}")
    df = df[df.label.isin(conf.classes)].copy()

    # build a lookup dict → index
    lookup = {tok: idx for idx, tok in enumerate(conf.classes)}
    # map to indices (will never produce NaN, because we pre-checked)
    df["label_idx"] = df["label"].map(lookup)

    # Ensure acquired datetime and year columns exist
    if "acquired" not in df.columns:
        if "source_tif" in df.columns:
            df["acquired"] = df["source_tif"].apply(lambda p: parse_dt_from_pth(Path(p)))
        else:
            raise ValueError(
                "load_labels: 'acquired' column missing and cannot derive from 'source_tif'."
            )
    df["acquired"] = pd.to_datetime(df["acquired"], errors="coerce")
    if df["acquired"].isna().any():
        bad = df.loc[df["acquired"].isna(), :].shape[0]

        logger.warning(f"Dropping {bad} rows with invalid 'acquired' timestamps")
        df = df.dropna(subset=["acquired"]).copy()

    return df


def create_splits(
    conf: EstuaryConfig, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits.

    Supported strategies:
      - "year": hold out one year for test and one for validation (manual),
                 optional union with a full region holdout for test.
      - "random": legacy stratified random split (fallback).
    """
    df = load_labels(conf)
    region_splits = pd.read_csv(conf.region_splits, index_col="region")

    df_train = df[df.region.isin(region_splits[region_splits.is_train].index)]
    df_val = df[df.region.isin(region_splits[region_splits.is_val].index)]
    df_test = df[df.region.isin(region_splits[region_splits.is_test].index)]

    # Log split sizes
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(df_train), len(df_val), len(df_test)
    )
    if verbose:
        logger.info("Train Class Split")
        logger.info(df_train["label_idx"].value_counts())
        logger.info("Val Class Split")
        logger.info(df_val["label_idx"].value_counts())
    return df_train, df_val, df_test


def calc_class_weights(conf: EstuaryConfig) -> tuple[float, ...]:
    df, _, _ = create_splits(conf, verbose=False)

    counts = df["label_idx"].value_counts().sort_index()
    # counts is a Series([n0, n1, n2]), where ni = number of examples in class i

    # total number of samples
    N = counts.sum()

    # number of classes
    C = len(counts)

    # weight for each class = N / (C * ni)
    weights = N / (C * counts.values)  # type: ignore

    return tuple(weights.tolist())


def parse_dt_from_pth(pth: Path) -> datetime.datetime:
    """Parse acquisition datetime from file stem prefix YYYYMMDD_HHMMSS_*"""
    datetime_str = "_".join(pth.stem.split("_")[:2])
    date_format = "%Y%m%d_%H%M%S"
    return datetime.datetime.strptime(datetime_str, date_format)


def load_tif(pth: Path, config: EstuaryConfig) -> tuple[np.ndarray, tuple[float, float]]:
    """Read SR GeoTIFF, return array, centroid (lat, lon).

    The returned array shape is (C,H,W) for subsequent torch transforms.
    """
    with rasterio.open(pth) as src:
        bounds = src.bounds
        crs = src.crs
        transformer = Transformer.from_crs(crs, 4326, always_xy=True)
        # centroid in lon/lat then lat/lon ordering
        centroid_x = (bounds.left + bounds.right) / 2
        centroid_y = (bounds.top + bounds.bottom) / 2
        lon, lat = transformer.transform(centroid_x, centroid_y)

        data = src.read(out_dtype=np.float32)
        nodata = src.read(1, masked=True).mask

        if nodata.all():
            img = np.zeros((*nodata.shape, 3), dtype=np.uint8)

        else:
            bands = config.bands.band_order(len(data))
            if config.bands == Bands.EIGHT and len(data) == 4:
                img = np.zeros((8, *nodata.shape), dtype=data.dtype)
                for i, b in enumerate(reversed(bands)):
                    img[b] = data[i]
            else:
                img = np.array([data[b] for b in bands])

        return img, (lon, lat)


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


def load_normalization(conf: EstuaryConfig) -> NormalizationStats:
    assert conf.normalization_path is not None
    with open(conf.normalization_path) as f:
        stats = json.load(f)
        power_scale = bool(stats["power_scale"])
        max_pixel_value = int(stats["max_raw_pixel_value"])

        idxes = conf.bands.eight_band_idxes()
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


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class EstuaryDataset(Dataset):
    """Loads PlanetScope SR data and builds a datacube."""

    def __init__(
        self,
        df: pd.DataFrame,
        conf: EstuaryConfig,
        train: bool = False,
    ) -> None:
        """
        Args
        ----
        df : DataFrame with columns ["region", "source_tif", "label", ...]
        conf: The train/inference config
        train : whether to apply train-time flips
        """
        self.df = df.reset_index(drop=True)
        self.conf = conf
        self.train = train
        self.norm_stats = load_normalization(conf)

        if train:
            self.resize_aug = K.RandomResizedCrop(
                (self.conf.train_size, self.conf.train_size),
                scale=self.conf.scale,
                resample=Resample.BICUBIC,
            )
        else:
            self.resize_aug = K.Resize(
                size=(conf.val_size, conf.val_size), resample=Resample.BICUBIC, antialias=True
            )

        augs: list[Any] = []
        if self.norm_stats.power_scale:
            augs.append(
                PowerTransformTorch(
                    self.norm_stats.lambdas.tolist(), self.norm_stats.max_pixel_value
                )
            )
        else:
            augs.append(ScaleNormalization(self.norm_stats.max_pixel_value))

        if train:
            augs += [
                K.RandomVerticalFlip(p=self.conf.vertical_flip_p),
                K.RandomHorizontalFlip(p=self.conf.horizontal_flip_p),
                K.RandomRotation90((0, 3), p=conf.rotation_p),
                K.RandomBrightness(
                    brightness=(max(0, 1 - conf.brightness), 1 + conf.brightness),
                    p=conf.brightness_p,
                ),
                K.RandomContrast(
                    contrast=(max(0, 1 - conf.contrast), 1 + conf.contrast),
                    p=conf.contrast_p,
                ),
                K.RandomSharpness(sharpness=self.conf.sharpness, p=self.conf.sharpness_p),
                K.RandomGaussianNoise(
                    mean=self.conf.gauss_mean,
                    std=self.conf.gauss_std,
                    p=self.conf.gauss_p,
                ),
                K.RandomGaussianBlur(
                    kernel_size=conf.blur_kernel_size,
                    sigma=conf.blur_sigma,
                    p=conf.blur_p,
                ),
                RandomChannelShift(
                    shift_limit=self.conf.channel_shift_limit,
                    num_channels=conf.bands.num_channels(),
                    p=self.conf.channel_shift_p,
                ),
                K.RandomErasing(scale=self.conf.erasing_scale, p=self.conf.erasing_p),
            ]

        augs += [
            K.Resize(size=(conf.val_size, conf.val_size)),
            K.Normalize(mean=self.norm_stats.mean.tolist(), std=self.norm_stats.std.tolist()),
        ]
        self.transforms = K.AugmentationSequential(*augs, data_keys=None)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return K.Denormalize(mean=self.norm_stats.mean.tolist(), std=self.norm_stats.std.tolist())(
            x
        )

    def __len__(self) -> int:
        if self.conf.debug:
            return min(self.conf.batch_size * 8, len(self.df))
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        tif_path = Path(row["source_tif"])

        if not tif_path.exists():
            raise FileNotFoundError(tif_path)

        data, _ = load_tif(tif_path, self.conf)
        pixels = torch.from_numpy(data.astype(np.float32))
        # Resize can create negative values :(
        pixels = self.resize_aug(pixels).clip(0, self.norm_stats.max_pixel_value)
        # Some Kornia per-sample augs (e.g., RandomResizedCrop) can return (1,C,H,W) for a 3D input.
        # Squeeze a possible leading singleton batch dim so DataLoader collates to (B,C,H,W) and
        # not (B,1,C,H,W).
        if pixels.ndim == 4 and pixels.shape[0] == 1:
            pixels = pixels.squeeze(0)
        label = torch.tensor(row["label_idx"], dtype=torch.long)

        return {"image": pixels, "label": label, "source_tif": str(tif_path)}


# -------------------------------------------------
# Lightning DataModule
# -------------------------------------------------
class EstuaryDataModule(LightningDataModule):
    """train/val/test splits with stratified shuffle on the `label` column."""

    def __init__(
        self,
        conf: EstuaryConfig,
    ) -> None:
        super().__init__()
        self.conf = conf

        self.save_hyperparameters(conf)

        # placeholders
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

        self.train_aug: K.AugmentationSequential | None = None
        self.val_aug: K.AugmentationSequential | None = None
        self.test_aug: K.AugmentationSequential | None = None

    # -------- Lightning hooks --------
    def prepare_data(self) -> None:
        # verify files exist & readable
        if not self.conf.data.exists():
            raise FileNotFoundError(self.conf.data)
        if not self.conf.metadata_path.exists():
            raise FileNotFoundError(self.conf.metadata_path)

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            df_train, df_val, df_test = create_splits(self.conf)

            # build datasets
            self.train_ds = EstuaryDataset(
                df=df_train,
                conf=self.conf,
                train=True,
            )
            self.train_aug = self.train_ds.transforms
            self.val_ds = EstuaryDataset(
                df=df_val,
                conf=self.conf,
                train=False,
            )
            self.val_aug = self.val_ds.transforms
            self.test_ds = EstuaryDataset(
                df=df_test,
                conf=self.conf,
                train=False,
            )
            self.test_aug = self.test_ds.transforms

    # -------- DataLoaders --------
    def train_dataloader(self):
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=num_workers(self.conf),
            pin_memory=self.conf.pin_memory,
            persistent_workers=self.conf.persistent_workers,
            prefetch_factor=self.conf.prefetch_factor if self.conf.prefetch_factor else None,
        )

    def val_dataloader(self):
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=num_workers(self.conf),
            pin_memory=self.conf.pin_memory,
            persistent_workers=self.conf.persistent_workers,
        )

    def test_dataloader(self):
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=num_workers(self.conf),
            pin_memory=self.conf.pin_memory,
            persistent_workers=self.conf.persistent_workers,
        )

    """
    These two functions are a hack to run augmentation on CPU when the device is apple MPS
    and on GPU otherwise.
    """

    def _aug_batch(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing:
                aug = self.test_aug
            elif self.trainer.predicting:
                aug = self.test_aug
            else:
                aug = self.test_aug

            assert aug is not None
            batch = aug(batch)
        return batch

    def on_before_batch_transfer(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        """Apply batch augmentations to the batch BEFORE it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            device: The device
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer and torch.backends.mps.is_available():
            batch = self._aug_batch(batch, dataloader_idx)
        return batch

    def on_after_batch_transfer(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        """Apply batch augmentations to the batch AFTER it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer and not torch.backends.mps.is_available():
            batch = self._aug_batch(batch, dataloader_idx)
        return batch
