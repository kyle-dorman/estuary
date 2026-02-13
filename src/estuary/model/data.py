import logging
from collections.abc import Iterable
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
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from estuary.model.config import (
    EstuaryConfig,
)
from estuary.util.bands import Bands
from estuary.util.data import cpu_count, load_normalization, parse_dt_from_pth
from estuary.util.transforms import (
    PowerTransformTorch,
    ScaleNormalization,
    basic_train_augs,
    unsure_train_augs,
)

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


def load_labels(conf: EstuaryConfig) -> pd.DataFrame:
    return _load_labels(conf.classes, conf.data)


def _load_labels(classes: Iterable[str], data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    df["orig_label"] = df.label

    if "perched open" not in classes:
        df.loc[df.label == "perched open", "label"] = "open"

    cls_diff = set(df.label.unique()) - set(classes)
    if cls_diff:
        logger.warning(f"Some label classes will be ignored {cls_diff}")
    df = df[df.label.isin(classes)].copy()

    # build a lookup dict → index
    lookup = {tok: idx for idx, tok in enumerate(classes)}
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
    """Create train/val/test splits. Method from EstuaryConfig.split_method

    Supported strategies:
      - "region": Read val/test holdout regions names from EstuaryConfig.region_splits
      - "crossval": Run a cross validation. Get index from EstuaryConfig.cv_index
            and num cross validations from EstuaryConfig.cv_folds
      - "yearly": Split by calendar year using EstuaryConfig.val_year and EstuaryConfig.test_year
    """
    df = load_labels(conf)

    logger.info(f"Splitting dataset using {conf.split_method}")

    if conf.split_method == "crossval":
        assert conf.cv_index < conf.cv_folds

        sgkf = StratifiedGroupKFold(n_splits=conf.cv_folds, shuffle=True, random_state=conf.seed)
        found = False
        for i, (train_idx, val_idx) in enumerate(
            sgkf.split(df, y=df["label_idx"], groups=df["region"])
        ):
            if i == conf.cv_index:
                found = True
                df_train = df.iloc[train_idx].copy()
                df_val = df.iloc[val_idx].copy()
                df_test = df_val.copy()
                break
        if not found:
            raise ValueError(f"cv_index {conf.cv_index} out of range for n_splits={conf.cv_folds}")
    elif conf.split_method == "region":
        region_splits = pd.read_csv(conf.region_splits, index_col="region")

        df_train = df[df.region.isin(region_splits[region_splits.is_train].index)].copy()
        df_val = df[df.region.isin(region_splits[region_splits.is_val].index)].copy()
        df_test = df[df.region.isin(region_splits[region_splits.is_test].index)].copy()
    elif conf.split_method == "yearly":
        # Yearly split: require val_year and test_year, and split by year column
        df["year"] = df["acquired"].dt.year  # type: ignore
        if conf.val_year is None or conf.test_year is None:
            raise ValueError("yearly split requires conf.val_year and conf.test_year")
        df_val = df[df["year"] == conf.val_year].copy()
        df_test = df[df["year"] == conf.test_year].copy()
        df_train = df[~df["year"].isin([conf.val_year, conf.test_year])].copy()
        if df_val.empty:
            raise ValueError(f"No samples found for val_year={conf.val_year}")
        if df_test.empty:
            raise ValueError(f"No samples found for test_year={conf.test_year}")
    else:
        raise RuntimeError(f"Unexpected split_method {conf.split_method}")

    # Log split sizes
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(df_train), len(df_val), len(df_test)
    )
    df_train["dataset"] = "train"
    df_val["dataset"] = "val"
    df_test["dataset"] = "test"
    if verbose:
        logger.info("Train Class Split")
        logger.info(df_train["label_idx"].value_counts().sort_index())
        logger.info("Val Class Split")
        logger.info(df_val["label_idx"].value_counts().sort_index())
        logger.info("Test Class Split")
        logger.info(df_test["label_idx"].value_counts().sort_index())
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
        nodata = src.read_masks(1) == 0

        if nodata.all():
            img = np.zeros((3, *nodata.shape), dtype=np.uint8)

        else:
            bands = config.bands.band_order(len(data))
            if config.bands == Bands.EIGHT and len(data) == 4:
                img = np.zeros((8, *nodata.shape), dtype=data.dtype)
                for i, b in enumerate(reversed(bands)):
                    img[b] = data[i]
            else:
                img = np.array([data[b] for b in bands])

        return img, (lon, lat)


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class EstuaryDataset(Dataset):
    """Loads PlanetScope SR data and builds a datacube."""

    def __init__(self, df: pd.DataFrame, conf: EstuaryConfig, train: bool = False) -> None:
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
        assert conf.normalization_path is not None
        self.norm_stats = load_normalization(conf.normalization_path, conf.bands)

        if train:
            self.resize_aug = K.RandomResizedCrop(
                (self.conf.train_size, self.conf.train_size),
                scale=self.conf.aug.scale,
                resample=Resample.BICUBIC,
            )
        else:
            self.resize_aug = K.Resize(
                size=(conf.val_size, conf.val_size), resample=Resample.BICUBIC, antialias=True
            )

        if self.norm_stats.power_scale:
            self.scale_aug = PowerTransformTorch(
                self.norm_stats.lambdas.tolist(), self.norm_stats.max_pixel_value
            )
        else:
            self.scale_aug = ScaleNormalization(self.norm_stats.max_pixel_value)

        self.scale_resize = K.AugmentationSequential(
            self.resize_aug, self.scale_aug, data_keys=None
        )

        if self.train:
            self._confident_tranfroms = K.AugmentationSequential(
                *basic_train_augs(self.conf.aug, self.conf.bands), data_keys=None
            )
            self._unsure_transforms = K.AugmentationSequential(
                *unsure_train_augs(self.conf.aug, self.conf.bands), data_keys=None
            )

        self.transforms = K.AugmentationSequential(
            K.Resize(size=(conf.val_size, conf.val_size)),
            K.Normalize(mean=self.norm_stats.mean.tolist(), std=self.norm_stats.std.tolist()),
            data_keys=None,
        )

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.norm_stats.mean.tolist()
        std = self.norm_stats.std.tolist()
        return K.Denormalize(mean=mean, std=std)(x)

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

        pixels = self.scale_resize({"image": pixels})["image"]
        if self.train:
            if self.conf.aug_level == "high":
                pixels = self._confident_tranfroms({"image": pixels})["image"]
            elif self.conf.aug_level == "low":
                pixels = self._unsure_transforms({"image": pixels})["image"]
            else:
                if row["source"] == "low_quality":
                    pixels = self._unsure_transforms({"image": pixels})["image"]
                else:
                    pixels = self._confident_tranfroms({"image": pixels})["image"]

        pixels = self.transforms({"image": pixels})["image"]

        # Some Kornia per-sample augs (e.g., RandomResizedCrop) can return (1,C,H,W) for a 3D input.
        # Squeeze a possible leading singleton batch dim so DataLoader collates to (B,C,H,W) and
        # not (B,1,C,H,W).
        if pixels.ndim == 4 and pixels.shape[0] == 1:
            pixels = pixels.squeeze(0)

        orig_label_str = row["orig_label"]
        label = torch.tensor(row["label_idx"], dtype=torch.long)

        return {
            "image": pixels,
            "label": label,
            "orig_label": orig_label_str,
            "source_tif": str(tif_path),
            "region": row["region"],
        }


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

        # self.train_aug: K.AugmentationSequential | None = None
        # self.val_aug: K.AugmentationSequential | None = None
        # self.test_aug: K.AugmentationSequential | None = None

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
            self.val_ds = EstuaryDataset(
                df=df_val,
                conf=self.conf,
                train=False,
            )
            self.test_ds = EstuaryDataset(
                df=df_test,
                conf=self.conf,
                train=False,
            )

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

    # """
    # These two functions are a hack to run augmentation on CPU when the device is apple MPS
    # and on GPU otherwise.
    # """

    # def _aug_batch(
    #     self, batch: dict[str, torch.Tensor], dataloader_idx: int
    # ) -> dict[str, torch.Tensor]:
    #     if self.trainer:
    #         if self.trainer.training:
    #             aug = self.train_aug
    #         elif self.trainer.validating or self.trainer.sanity_checking:
    #             aug = self.val_aug
    #         elif self.trainer.testing:
    #             aug = self.test_aug
    #         elif self.trainer.predicting:
    #             aug = self.test_aug
    #         else:
    #             aug = self.test_aug

    #         assert aug is not None
    #         batch = aug(batch)
    #     return batch

    # def on_before_batch_transfer(
    #     self, batch: dict[str, torch.Tensor], dataloader_idx: int
    # ) -> dict[str, torch.Tensor]:
    #     """Apply batch augmentations to the batch BEFORE it is transferred to the device.

    #     Args:
    #         batch: A batch of data that needs to be altered or augmented.
    #         device: The device
    #         dataloader_idx: The index of the dataloader to which the batch belongs.

    #     Returns:
    #         A batch of data.
    #     """
    #     if self.trainer and torch.backends.mps.is_available():
    #         batch = self._aug_batch(batch, dataloader_idx)
    #     return batch

    # def on_after_batch_transfer(
    #     self, batch: dict[str, torch.Tensor], dataloader_idx: int
    # ) -> dict[str, torch.Tensor]:
    #     """Apply batch augmentations to the batch AFTER it is transferred to the device.

    #     Args:
    #         batch: A batch of data that needs to be altered or augmented.
    #         dataloader_idx: The index of the dataloader to which the batch belongs.

    #     Returns:
    #         A batch of data.
    #     """
    #     if self.trainer and not torch.backends.mps.is_available():
    #         batch = self._aug_batch(batch, dataloader_idx)
    #     return batch
