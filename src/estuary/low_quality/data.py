import logging
from pathlib import Path
from typing import Any

import kornia.augmentation as K
import numpy as np
import pandas as pd
import rasterio
import torch
from kornia.constants import Resample
from lightning.pytorch import LightningDataModule
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, Dataset

from estuary.low_quality.config import QualityConfig
from estuary.util.bands import Bands
from estuary.util.data import cpu_count, load_normalization, parse_dt_from_pth
from estuary.util.transforms import RandomChannelShift

logger = logging.getLogger(__name__)


def num_workers(conf: QualityConfig) -> int:
    # number of CUDA devices
    nd = max(1, conf.world_size)
    per_gpu_count = cpu_count() // nd

    if conf.workers == -1:
        return per_gpu_count
    # number of workers
    nw = min([per_gpu_count, conf.workers])

    return nw


def load_labels(conf: QualityConfig) -> pd.DataFrame:
    return _load_labels(conf.data)


def _load_labels(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    df = df[df.cluster_label >= 0].copy()

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


def create_splits(conf: QualityConfig, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/val splits."""
    rng = np.random.RandomState(conf.seed)

    df = load_labels(conf)

    logger.info("Splitting dataset")

    cloudy_df = df[df.cluster_label == 1]
    good_df = df[df.cluster_label == 0]

    val_idxes = rng.choice(good_df.index, len(cloudy_df), replace=False)

    df_val = pd.concat([cloudy_df, good_df.loc[val_idxes]]).copy()
    df_train = good_df[~good_df.index.isin(val_idxes)].copy()

    # Log split sizes
    logger.info("Split sizes -> train: %d, val: %d", len(df_train), len(df_val))
    df_train["dataset"] = "train"
    df_val["dataset"] = "val"
    if verbose:
        logger.info("Train Class Split")
        logger.info(df_train["cluster_label"].value_counts().sort_index())
        logger.info("Val Class Split")
        logger.info(df_val["cluster_label"].value_counts().sort_index())
    return df_train, df_val


def load_tif(pth: Path, config: QualityConfig) -> np.ndarray:
    """Read SR GeoTIFF, return array, centroid (lat, lon).

    The returned array shape is (C,H,W) for subsequent torch transforms.
    """
    with rasterio.open(pth) as src:
        data = src.read(out_dtype=np.float32)
        nodata = src.read_masks(1) == 0

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

        return img


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class LowQualityDataset(Dataset):
    """Loads PlanetScope SR data and builds a datacube."""

    def __init__(
        self,
        df: pd.DataFrame,
        conf: QualityConfig,
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

        # Create normaliztion
        assert conf.normalization_path is not None
        self.norm_stats = load_normalization(conf.normalization_path, conf.bands)
        scale = np.array([self.norm_stats.max_pixel_value] * self.conf.bands.num_channels())
        st = StandardScaler(with_mean=False)
        if self.norm_stats.power_scale:
            pt = PowerTransformer(standardize=False)
            pt.lambdas_ = self.norm_stats.lambdas
            max_value = pt.transform(scale[None])[0]
            st.scale_ = max_value
            self.norm = Pipeline([("PowerTransformer", pt), ("StandardScaler", st)])
        else:
            st.scale_ = scale
            self.norm = Pipeline([("StandardScaler", st)])

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
                RandomChannelShift(
                    shift_limit=self.conf.channel_shift_limit,
                    num_channels=conf.bands.num_channels(),
                    p=self.conf.channel_shift_p,
                ),
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

        data = load_tif(tif_path, self.conf)
        shp = data.shape
        norm_data = self.norm.transform(data.reshape(len(data), -1).T).T.reshape(shp)

        if self.train:
            norm_data = self.perturb_image(norm_data)

        pixels = torch.from_numpy(norm_data.astype(np.float32))
        # Resize can create negative values :(
        pixels = self.resize_aug(pixels).clip(0, self.norm_stats.max_pixel_value)
        # Some Kornia per-sample augs (e.g., RandomResizedCrop) can return (1,C,H,W) for a 3D input.
        # Squeeze a possible leading singleton batch dim so DataLoader collates to (B,C,H,W) and
        # not (B,1,C,H,W).
        if pixels.ndim == 4 and pixels.shape[0] == 1:
            pixels = pixels.squeeze(0)

        label = torch.tensor(row["label_idx"], dtype=torch.long)

        return {
            "image": pixels,
            "label": label,
            "source_tif": str(tif_path),
            "region": row["region"],
        }

    def perturb_image(self, data: np.ndarray) -> np.ndarray:
        return data


# -------------------------------------------------
# Lightning DataModule
# -------------------------------------------------
class LowQualityDataModule(LightningDataModule):
    """train/val/test splits with stratified shuffle on the `label` column."""

    def __init__(
        self,
        conf: QualityConfig,
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
        if self.conf.normalization_path is not None and not self.conf.normalization_path.exists():
            raise FileNotFoundError(self.conf.normalization_path)

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            df_train, df_val = create_splits(self.conf)

            # build datasets
            self.train_ds = LowQualityDataset(
                df=df_train,
                conf=self.conf,
                train=True,
            )
            self.train_aug = self.train_ds.transforms
            self.val_ds = LowQualityDataset(
                df=df_val,
                conf=self.conf,
                train=False,
            )
            self.val_aug = self.val_ds.transforms

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
