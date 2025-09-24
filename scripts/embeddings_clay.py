import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterio
import torch
import tqdm
from pyproj import Transformer
from torchvision.transforms import v2

from estuary.clay.data import (
    load_planetscope_sr4_metadata,
    load_planetscope_sr8_metadata,
    prep_datacube,
)
from estuary.model.data import parse_dt_from_pth

# -----------------------------
# Constants / defaults
# -----------------------------
DEFAULT_METADATA_YAML = "/Users/kyledorman/data/models/clay/metadata.yaml"
DEFAULT_ENCODER_PATH = "/Users/kyledorman/data/models/clay/clay-v1.5-encoder_256.pt2"

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("clay.embed")


def _setup_logging():
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -----------------------------
# Metadata & helpers
# -----------------------------
def load_tif_crop(pth: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Read SR GeoTIFF return array, centroid (lat, lon)

    The returned array shape is (C,H,W) for subsequent torch transforms.
    """
    with rasterio.open(pth) as src:
        if src.count != 4:
            raise RuntimeError(f"Expected 4 bands in {pth}, found {src.count}")

        bounds = src.bounds
        crs = src.crs
        transformer = Transformer.from_crs(crs, 4326, always_xy=True)
        # centroid in lon/lat then lat/lon ordering
        centroid_x = (bounds.left + bounds.right) / 2
        centroid_y = (bounds.top + bounds.bottom) / 2
        lon, lat = transformer.transform(centroid_x, centroid_y)

        data = src.read()
        return data, (lat, lon)


def load_encoder(encoder_path: Path):
    """Load the exported Torch encoder (CPU only)."""
    mod = torch.export.load(str(encoder_path)).module()
    mod.eval()
    return mod


def encode_one(
    encoder,
    band_meta: dict,
    tif_path: Path,
    transforms: v2.Compose,
) -> np.ndarray:
    data, (lat, lon) = load_tif_crop(tif_path)
    dt = parse_dt_from_pth(tif_path)
    waves = torch.tensor(
        [band_meta["bands"]["wavelength"][b] * 1000 for b in band_meta["band_order"]],
        dtype=torch.float32,
    )
    gsd = torch.Tensor([band_meta["gsd"]])
    datacube = prep_datacube(data, lat, lon, dt, gsd, waves.numpy(), transforms)
    with torch.no_grad():
        emb = encoder(datacube).detach().cpu().numpy()[0]
    return emb


def save_embedding(save_dir: Path, key: str, emb: np.ndarray, overwrite: bool) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{key}.npy"
    if out.exists() and not overwrite:
        return out
    np.save(out, emb)
    return out


# -----------------------------
# CLI
# -----------------------------
@click.command()
@click.option(
    "--metadata-yaml",
    type=click.Path(path_type=Path),
    default=Path(DEFAULT_METADATA_YAML),
    show_default=True,
    help="Path to metadata.yaml with PlanetScope SR stats.",
)
@click.option(
    "--encoder-path",
    type=click.Path(path_type=Path),
    default=Path(DEFAULT_ENCODER_PATH),
    show_default=True,
    help="Path to exported CLAY encoder .pt2 file.",
)
@click.option(
    "--labels-path",
    type=click.Path(path_type=Path, exists=True, file_okay=True),
    required=True,
    help="labels.csv file",
)
@click.option(
    "--save-path",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Where to day the encodings",
)
@click.option(
    "--size",
    type=int,
    default=256,
    show_default=True,
    help="Resize target (pixels) for encoder input; must match model training size.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing .npy embeddings.")
def main(
    metadata_yaml: Path,
    encoder_path: Path,
    labels_path: Path,
    save_path: Path,
    size: int,
    overwrite: bool,
):
    """Embed all image regions using the CLAY encoder (CPU)."""
    _setup_logging()

    # Fail fast on required resources
    if not metadata_yaml.exists():
        raise FileNotFoundError(f"Missing metadata YAML: {metadata_yaml}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing encoder file: {encoder_path}")

    labels = pd.read_csv(labels_path)

    # Load metadata & encoder
    md4 = load_planetscope_sr4_metadata(metadata_yaml)
    md8 = load_planetscope_sr8_metadata(metadata_yaml)
    encoder = load_encoder(encoder_path)

    for sensor, band_meta in zip(["dove", "superdove"], [md4, md8], strict=False):
        mean = [band_meta["bands"]["mean"][b] for b in band_meta["band_order"]]
        std = [band_meta["bands"]["std"][b] for b in band_meta["band_order"]]

        transforms = v2.Compose(
            [
                v2.Resize(size=(size, size), interpolation=3),
                v2.Normalize(mean=mean, std=std),
            ]
        )

        for region in tqdm.tqdm(labels.region.unqiue()):
            rdf = labels[(labels.region == region) & (labels.dove == sensor)]
            save_dir = save_path / region

            for _, row in rdf.iterrows():
                source_tif = Path(row.source_tif)
                # Compute embedding
                emb = encode_one(encoder, band_meta, source_tif, transforms)
                save_embedding(save_dir, source_tif.stem, emb, overwrite=overwrite)

    logger.info("Done.")


if __name__ == "__main__":
    main()
