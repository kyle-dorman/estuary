import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from estuary.model.config import Bands, EstuaryConfig
from estuary.model.data import create_splits, load_tif, normalize_timestamp, num_workers
from scripts.embeddings_clay import normalize_latlon

logger = logging.getLogger(__name__)


def load_planetscope_sr8_metadata(metadata_yaml: Path) -> dict:
    """Load sensor metadata for 8-channel PlanetScope SR profile.

    Returns a dict like metadata["planetscope-sr"] with keys:
      - band_order
      - rgb_indices
      - gsd
      - bands{"mean"|"std"|"wavelength"}[band_name]
    """
    with open(metadata_yaml) as f:
        metadata = yaml.safe_load(f)

    return metadata["planetscope-sr"]


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

    md4 = {
        "band_order": channel_4_band_order,
        "rgb_indices": [3, 2, 1],
        "gsd": planetscope["gsd"],
        "bands": {},
    }

    # Filter band stats to the 4 channels
    bands = {}
    for k, vs in planetscope["bands"].items():
        vs4 = {kk: vv for kk, vv in vs.items() if kk in channel_4_band_order}
        bands[k] = vs4
    md4["bands"] = bands
    return md4


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
class ClayEstuaryDataset(Dataset):
    """Loads PlanetScope SR crops and builds a datacube."""

    def __init__(
        self,
        df: pd.DataFrame,
        conf: EstuaryConfig,
        tif_root: Path | None = None,
        train: bool = False,
    ) -> None:
        """
        Args
        ----
        df : DataFrame with columns ["region", "source_tif", "label", ...]
        conf: Run config
        tif_root : optional prefix to prepend to source_tif
        train : whether to apply train-time flips
        """
        self.df = df.reset_index(drop=True)
        self.tif_root = Path(tif_root) if tif_root else None
        self.conf = conf
        self.train = train

        # --- build transforms ---
        if conf.bands == Bands.EIGHT:
            band_meta = load_planetscope_sr8_metadata(conf.metadata_path)
        else:
            band_meta = load_planetscope_sr4_metadata(conf.metadata_path)

        mean = [band_meta["bands"]["mean"][b] for b in band_meta["band_order"]]
        std = [band_meta["bands"]["std"][b] for b in band_meta["band_order"]]
        self.gsd = torch.Tensor([band_meta["gsd"]])

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
            [band_meta["bands"]["wavelength"][b] * 1000 for b in band_meta["band_order"]],
            dtype=torch.float32,
        )

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

        data, (lat, lon) = load_tif(tif_path, self.conf)

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
class ClayEstuaryDataModule(LightningDataModule):
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
        if not self.conf.metadata_path.exists():
            raise FileNotFoundError(self.conf.metadata_path)

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            df_train, df_val, df_test = create_splits(self.conf)

            # build datasets
            self.train_ds = ClayEstuaryDataset(
                df=df_train,
                conf=self.conf,
                train=True,
            )
            self.val_ds = ClayEstuaryDataset(
                df=df_val,
                conf=self.conf,
                train=False,
            )
            self.test_ds = ClayEstuaryDataset(
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
