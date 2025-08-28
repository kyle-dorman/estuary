import datetime
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from lightning.pytorch import LightningDataModule
from pyproj import Transformer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from estuary.clay.config import EstuaryConfig

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
    df["year"] = df["acquired"].dt.year.astype(int)

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

    if verbose:
        # Log counts per year for transparency
        year_label_counts = df.groupby(["year", "label"]).size().sort_index()
        logger.info("Samples per year/label:\n" + year_label_counts.to_string())

    def _check(name: str, d: pd.DataFrame) -> None:
        if len(d) < conf.min_rows_per_split:
            raise ValueError(f"{name} split too small: {len(d)} < {conf.min_rows_per_split}")
        if conf.require_all_classes:
            present = set(d["label_idx"].unique().tolist())
            needed = set(range(len(conf.classes)))
            if not needed.issubset(present):
                missing = sorted(needed - present)
                raise ValueError(
                    f"{name} split missing class indices {missing}; please adjust years/region."
                )

    if conf.test_year is None and conf.val_year is None:
        years = df.year.unique()
        years.sort()
        test_year = years[-1]
        val_year = years[-2]
    elif conf.test_year is None and conf.val_year is None:
        test_year = conf.test_year
        val_year = conf.val_year
    else:
        raise ValueError(
            "Split requires both conf.test_year and conf.val_year to be set or neither."
        )

    if test_year == val_year:
        raise ValueError("conf.test_year and conf.val_year must be different.")

    mask_test_year = df["year"] == test_year
    mask_val_year = df["year"] == val_year
    mask_holdout = (
        (df["region"] == conf.holdout_region)
        if conf.holdout_region
        else pd.Series(False, index=df.index)
    )

    # Test is union of test_year and entire holdout_region (if provided)
    test_mask = mask_test_year | mask_holdout
    val_mask = mask_val_year & ~test_mask  # keep splits disjoint
    train_mask = ~(test_mask | val_mask)  # remainder

    df_test = df.loc[test_mask].copy()
    df_val = df.loc[val_mask].copy()
    df_train = df.loc[train_mask].copy()

    # Final guard: if holdout_region is set, ensure it's fully excluded from train/val
    if conf.holdout_region:
        assert conf.holdout_region not in df_train.region.unique()
        assert conf.holdout_region not in df_val.region.unique()

    # Checks
    _check("train", df_train)
    _check("val", df_val)
    _check("test", df_test)

    # Log split sizes
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(df_train), len(df_val), len(df_test)
    )
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


def load_planetscope_sr4_metadata(metadata_yaml: Path) -> dict:
    """Load sensor metadata and derive a 4-channel PlanetScope SR profile.

    Returns a dict like metadata["planetscope-sr-4"] with keys:
      - band_order
      - rgb_indices
      - gsd
      - bands{"mean"|"std"|"wavelength"}[band_name]
    """
    with open(metadata_yaml) as f:
        metadata = yaml.safe_load(f)

    channel_4_band_order = ["blue", "green", "red", "nir"]
    planetscope = metadata["planetscope-sr"]

    md4 = {"band_order": channel_4_band_order, "rgb_indices": [3, 2, 1], "gsd": 3, "bands": {}}

    # Filter band stats to the 4 channels
    bands = {}
    for k, vs in planetscope["bands"].items():
        vs4 = {kk: vv for kk, vv in vs.items() if kk in channel_4_band_order}
        bands[k] = vs4
    md4["bands"] = bands
    return md4


def parse_dt_from_pth(pth: Path) -> datetime.datetime:
    """Parse acquisition datetime from file stem prefix YYYYMMDD_HHMMSS_*"""
    datetime_str = "_".join(pth.stem.split("_")[:2])
    date_format = "%Y%m%d_%H%M%S"
    return datetime.datetime.strptime(datetime_str, date_format)


def load_tif_crop(
    pth: Path, crop: tuple[int, int, int, int], num_channels: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """Read 4-band SR GeoTIFF, crop to window, return array, centroid (lat, lon).

    The returned array shape is (C,H,W) for subsequent torch transforms.
    """
    with rasterio.open(pth) as src:
        if src.count != num_channels:
            raise RuntimeError(f"Expected 4 bands in {pth}, found {src.count}")

        bounds = src.bounds
        crs = src.crs
        transformer = Transformer.from_crs(crs, 4326, always_xy=True)
        # centroid in lon/lat then lat/lon ordering
        centroid_x = (bounds.left + bounds.right) / 2
        centroid_y = (bounds.top + bounds.bottom) / 2
        lon, lat = transformer.transform(centroid_x, centroid_y)

        start_w, start_h, end_w, end_h = crop
        data = src.read()[:, start_h:end_h, start_w:end_w]
        return data, (lat, lon)


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


def prep_datacube(
    image: np.ndarray,
    lat: float,
    lon: float,
    date: datetime.datetime,
    gsd: torch.Tensor,
    waves: np.ndarray,
    transforms: v2.Compose,
):
    """Prepare a dict of tensors for the encoder forward pass."""
    (week_sin, week_cos), (hour_sin, hour_cos) = normalize_timestamp(date)
    (lat_sin, lat_cos), (lon_sin, lon_cos) = normalize_latlon(lat, lon)

    pixels = torch.from_numpy(image.astype(np.float32))
    pixels = transforms(pixels)

    return {
        "pixels": pixels,
        "time": torch.tensor(
            [week_sin, week_cos, hour_sin, hour_cos],
            dtype=torch.float32,
        ),
        "latlon": torch.tensor([lat_sin, lat_cos, lon_sin, lon_cos], dtype=torch.float32),
        "waves": torch.tensor(
            waves,
        ),
        "gsd": gsd,
    }


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class EstuaryDataset(Dataset):
    """Loads PlanetScope SR 4-band crops and builds a datacube."""

    def __init__(
        self,
        df: pd.DataFrame,
        crops_map: dict[str, list[int]],
        conf: EstuaryConfig,
        tif_root: Path | None = None,
        train: bool = False,
    ) -> None:
        """
        Args
        ----
        df : DataFrame with columns ["region", "source_tif", "label", ...]
        crops_map : region-> [start_w, start_h, end_w, end_h]
        metadata_yaml : PlanetScope SR metadata yaml
        classes: the list of class strings
        tif_root : optional prefix to prepend to source_tif
        out_size : final H=W in pixels after resize
        train : whether to apply train-time flips
        bands: number of bands to use
        """
        self.df = df.reset_index(drop=True)
        self.crops_map = crops_map
        self.tif_root = Path(tif_root) if tif_root else None
        self.conf = conf
        self.train = train

        # --- build transforms ---
        md4 = load_planetscope_sr4_metadata(conf.metadata_path)
        mean = [md4["bands"]["mean"][b] for b in md4["band_order"]]
        std = [md4["bands"]["std"][b] for b in md4["band_order"]]
        self.gsd = torch.Tensor([md4["gsd"]])

        tfs: list[v2.Transform] = []
        if train:
            tfs += [
                v2.RandomHorizontalFlip(conf.horizontal_flip),
                v2.RandomVerticalFlip(conf.vertical_flip),
                v2.RandomResizedCrop(
                    size=(conf.chip_size, conf.chip_size),
                    scale=(conf.min_scale, 1.0),
                    antialias=True,
                ),
            ]
        tfs += [
            v2.Resize(size=(conf.chip_size, conf.chip_size), interpolation=3),
            v2.Normalize(mean=mean, std=std),
        ]
        self.transforms = v2.Compose(tfs)
        self.waves = torch.tensor(
            [md4["bands"]["wavelength"][b] * 1000 for b in md4["band_order"]],
            dtype=torch.float32,
        )

        # --- upfront validation of regions ---
        missing = [r for r in self.df["region"].unique() if r not in self.crops_map]
        if missing:
            raise RuntimeError(f"The following regions are missing in region_crops.json: {missing}")

    def __len__(self) -> int:
        if self.conf.debug:
            return min(self.conf.batch_size * 8, len(self.df))
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.df.iloc[idx]
        tif_path = Path(row["source_tif"])
        if self.tif_root and not tif_path.is_absolute():
            tif_path = self.tif_root / tif_path

        if not tif_path.exists():
            raise FileNotFoundError(tif_path)

        crop_box: tuple[int, int, int, int] = tuple(self.crops_map[row["region"]])  # type: ignore[arg-type]
        assert len(crop_box) == 4
        data, (lat, lon) = load_tif_crop(tif_path, crop_box, self.conf.bands)

        dt = row.acquired
        datacube = prep_datacube(
            image=data,
            lat=lat,
            lon=lon,
            date=dt,
            gsd=self.gsd,
            waves=self.waves.numpy(),  # helper expects np.ndarray
            transforms=self.transforms,
        )

        label = torch.tensor(row["label_idx"], dtype=torch.long)
        return datacube, label


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

    # -------- Lightning hooks --------
    def prepare_data(self) -> None:
        # verify files exist & readable
        if not self.conf.data.exists():
            raise FileNotFoundError(self.conf.data)
        if not self.conf.region_crops_json.exists():
            raise FileNotFoundError(self.conf.region_crops_json)
        if not self.conf.metadata_path.exists():
            raise FileNotFoundError(self.conf.metadata_path)

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            with open(self.conf.region_crops_json) as f:
                crops_map: dict[str, list[int]] = json.load(f)

            df_train, df_val, df_test = create_splits(self.conf)

            # build datasets
            self.train_ds = EstuaryDataset(
                df=df_train,
                crops_map=crops_map,
                conf=self.conf,
                train=True,
            )
            self.val_ds = EstuaryDataset(
                df=df_val,
                crops_map=crops_map,
                conf=self.conf,
                train=False,
            )
            self.test_ds = EstuaryDataset(
                df=df_test,
                crops_map=crops_map,
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
