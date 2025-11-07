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
from torch.utils.data import DataLoader, Dataset

from estuary.low_quality.config import QualityConfig
from estuary.util.bands import Bands
from estuary.util.data import cpu_count, load_normalization, parse_dt_from_pth
from estuary.util.transforms import (
    MisalignedImage,
    PowerTransformTorch,
    RandomPlasmaFog,
    ScaledRandomGaussianNoise,
    ScaleNormalization,
)

logger = logging.getLogger(__name__)


def make_blur_aug():
    return K.AugmentationSequential(
        K.RandomMotionBlur(
            kernel_size=9,
            angle=35,
            direction=0.5,
            border_type=2,
            p=1.0,
        ),
        K.RandomGaussianBlur(
            kernel_size=7,
            sigma=(1.0, 3.0),
            p=1.0,
        ),
        K.RandomBoxBlur(
            kernel_size=(7, 7),
            p=1.0,
        ),
        random_apply=1,
    )


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
    conf: QualityConfig, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val splits."""
    df_train = load_labels(conf)
    df_train = df_train[df_train.cluster_label >= 0].copy()

    df_test = _load_labels(conf.test_data)
    df_test["cluster_label"] = df_test.label.apply(lambda a: int(a == "unsure"))

    df_val = _load_labels(conf.val_data)
    df_val["cluster_label"] = df_val.label.apply(lambda a: int(a == "unsure"))

    test_images = df_test.source_tif.tolist()
    df_val = df_val[~df_val.source_tif.isin(test_images)].reset_index(drop=True)
    test_val_images = df_val.source_tif.tolist() + test_images

    df_train = df_train[~df_train.source_tif.isin(test_val_images)].reset_index(drop=True)

    logger.info("Splitting dataset")

    # Log split sizes
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(df_train), len(df_val), len(df_test)
    )
    df_train["dataset"] = "train"
    df_val["dataset"] = "val"
    if verbose:
        logger.info("Train Class Split")
        logger.info(df_train["cluster_label"].value_counts().sort_index())
        logger.info("Val Class Split")
        logger.info(df_val["cluster_label"].value_counts().sort_index())
        logger.info("Test Class Split")
        logger.info(df_test["cluster_label"].value_counts().sort_index())

    return df_train, df_val, df_test


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
        self.rng = np.random.default_rng(self.conf.seed)
        self.rng_t = torch.Generator()
        self.rng_t.manual_seed(self.conf.seed)

        # Create normaliztion
        assert conf.normalization_path is not None
        self.norm_stats = load_normalization(conf.normalization_path, conf.bands)

        if self.norm_stats.power_scale:
            self.norm = PowerTransformTorch(
                self.norm_stats.lambdas.tolist(), self.norm_stats.max_pixel_value
            )
        else:
            self.norm = ScaleNormalization(self.norm_stats.max_pixel_value)

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

        self.base_augs = K.AugmentationSequential(self.norm, self.resize_aug, data_keys=None)

        foggy = K.AugmentationSequential(
            K.RandomSharpness(sharpness=5.0, p=1.0),
            K.RandomPosterize(
                p=1.0,
                bits=(6, 7),
            ),
            ScaledRandomGaussianNoise(
                std=0.05,
                p=1.0,
            ),
            K.RandomPlasmaBrightness(p=1.0, intensity=(0.05, 0.1)),
            RandomPlasmaFog(
                p=1.0,
                fog_intensity=(0.8, 1.0),
                roughness=(0.5, 0.6),
            ),
        )

        foggy_color = K.AugmentationSequential(
            K.RandomPosterize(
                p=1.0,
                bits=(7, 7),
            ),
            RandomPlasmaFog(
                p=1.0,
                fog_intensity=(0.2, 0.2),
                roughness=(0.5, 0.7),
            ),
            K.RandomPlasmaBrightness(p=1.0, intensity=(0.6, 0.6), roughness=(0.5, 0.5)),
        )

        hazy_blur = K.AugmentationSequential(
            make_blur_aug(),
            RandomPlasmaFog(
                p=1.0,
                fog_intensity=(1.0, 1.0),
                roughness=(0.5, 0.6),
            ),
        )

        iso_noise = K.AugmentationSequential(
            make_blur_aug(),
            K.RandomGaussianNoise(
                std=0.15,
                p=1.0,
            ),
        )

        misalgined = K.AugmentationSequential(
            MisalignedImage(
                p=1.0,
                angle_deg=(-35.0, 35.0),
                edge_shift=(-10, 10),
                offset=(-20, 20),
                border_crop=30,
            ),
            K.Resize(
                size=(conf.val_size, conf.val_size), resample=Resample.BICUBIC, antialias=True
            ),
        )

        self.perturb_augs = K.AugmentationSequential(
            misalgined,
            iso_noise,
            hazy_blur,
            foggy,
            foggy_color,
            random_apply=1,
            data_keys=None,
        )

        augs: list[Any] = []
        if train:
            augs += [
                K.RandomVerticalFlip(p=self.conf.vertical_flip_p),
                K.RandomHorizontalFlip(p=self.conf.horizontal_flip_p),
                K.RandomRotation90((0, 3), p=conf.rotation_p),
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

        # Normalize to 0-1 and resize
        pixels = torch.from_numpy(data.astype(np.float32))
        pixels = self.base_augs({"image": pixels})["image"]

        if self.train:
            if self.rng.random() > 0.5:
                pixels = self.perturb_augs({"image": pixels})["image"]
                label = torch.tensor(1, dtype=torch.long)
            else:
                label = torch.tensor(row["cluster_label"], dtype=torch.long)
        else:
            label = torch.tensor(row["cluster_label"], dtype=torch.long)

        if pixels.ndim == 4 and pixels.shape[0] == 1:
            pixels = pixels.squeeze(0)

        return {
            "image": pixels,
            "label": label,
            "source_tif": str(tif_path),
            "region": row["region"],
        }


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
        if not self.conf.test_data.exists():
            raise FileNotFoundError(self.conf.test_data)
        if not self.conf.val_data.exists():
            raise FileNotFoundError(self.conf.val_data)
        if self.conf.normalization_path is not None and not self.conf.normalization_path.exists():
            raise FileNotFoundError(self.conf.normalization_path)

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            df_train, df_val, df_test = create_splits(self.conf)

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

            self.test_ds = LowQualityDataset(
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
