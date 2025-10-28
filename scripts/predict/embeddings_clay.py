import logging
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import click
import numpy as np
import rasterio
import torch
import tqdm.auto as tqdm
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
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
DEFAULT_ENCODER_PATH = "/Users/kyledorman/data/models/clay/clay-v1.5-encoder-cpu.pt2"

"""
uv run .venv/lib/python3.12/site-packages/claymodel/finetune/embedder/factory.py
    --ckpt_path /Users/kyledorman/data/models/clay/clay-v1.5.ckpt
    --device cpu
    --name clay-v1.5-encoder-cpu.pt2
    --ep
"""

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("clay.embed")


def _setup_logging(verbosity: int = 0):
    if logger.handlers:
        # Avoid adding duplicate handlers on repeated CLI runs
        return
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)


# -----------------------------
# Geo helpers
# -----------------------------
@lru_cache(maxsize=16)
def _to_wgs84_transformer(crs_str: str) -> Transformer:
    """Cache CRS->EPSG:4326 transformers (handles many tiles sharing the same CRS)."""
    return Transformer.from_crs(crs_str, 4326, always_xy=True)


def _centroid_latlon(src: DatasetReader) -> tuple[float, float]:
    bounds = src.bounds
    transformer = _to_wgs84_transformer(src.crs.to_string())
    cx = (bounds.left + bounds.right) / 2
    cy = (bounds.top + bounds.bottom) / 2
    lon, lat = transformer.transform(cx, cy)
    return float(lat), float(lon)


# -----------------------------
# I/O + encoding
# -----------------------------


def read_sr_resize(pth: Path, size: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Read a PlanetScope SR GeoTIFF, resampling on read to (size,size).

    Returns:
        arr: (C, size, size) float32
        (lat, lon): centroid in degrees
    """
    with rasterio.open(pth) as src:
        # Resample on read to avoid loading full-res and resizing later
        out_shape = (src.count, size, size)
        data = src.read(
            out_shape=out_shape,
            resampling=Resampling.bilinear,
            out_dtype="float32",
        )
        lat, lon = _centroid_latlon(src)
    return data, (lat, lon)


def load_encoder(encoder_path: Path):
    """Load the exported Torch encoder (CPU only)."""
    mod = torch.export.load(str(encoder_path)).module()
    return mod


def encode_one(
    encoder,
    band_meta: dict,
    tif_path: Path,
    size: int,
    norm: v2.Compose,
) -> np.ndarray:
    arr, (lat, lon) = read_sr_resize(tif_path, size=size)
    dt = parse_dt_from_pth(tif_path)
    # waves in nm
    waves = torch.tensor(
        [band_meta["bands"]["wavelength"][b] * 1000 for b in band_meta["band_order"]],
        dtype=torch.float32,
    )
    gsd = torch.Tensor([band_meta["gsd"]])

    # prep_datacube expects (C,H,W) float arrays and metadata
    datacube = prep_datacube(arr, lat, lon, dt, gsd, waves.numpy(), norm)
    for key in datacube.keys():
        if key in ["gsd", "waves"]:
            continue
        a: torch.Tensor = datacube[key]
        datacube[key] = a.unsqueeze(dim=0)
    with torch.inference_mode():
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
# Discovery
# -----------------------------


def discover_sr_paths(roots: Iterable[Path]) -> list[Path]:
    """Find all SR TIFFs under the provided roots."""
    paths: list[Path] = []
    for r in roots:
        # pattern captures 4- and 8-band SR names (e.g., *_SR_clip.tif, *_SR_8b_clip.tif)
        paths.extend(r.glob("*/*/*/files/*_SR*.tif"))
    # deterministic ordering
    return sorted(set(paths))


def group_by_region(paths: Iterable[Path]) -> dict[str, list[Path]]:
    """Group tif paths by numeric region id inferred from directory structure.

    Expects .../<year>/<month>/<region>/files/<asset>.tif
    """
    buckets: dict[str, list[Path]] = {}
    for p in paths:
        # parents: files -> region -> month -> year -> ...
        region_name = p.parents[1].name
        buckets.setdefault(region_name, []).append(p)
    return buckets


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
    "--save-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory where embeddings will be written as <region>/<stem>.npy",
)
@click.option(
    "--size",
    type=int,
    default=256,
    show_default=True,
    help="Spatial size (pixels) to resample inputs to before encoding.",
)
@click.option("-v", "--verbose", count=True, help="Increase log verbosity (-v).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing .npy embeddings.")
def main(
    metadata_yaml: Path,
    encoder_path: Path,
    save_dir: Path,
    size: int,
    verbose: int,
    overwrite: bool,
):
    """Embed all image regions using the CLAY encoder (CPU)."""
    _setup_logging(verbose)

    # Fail fast on required resources
    if not metadata_yaml.exists():
        raise FileNotFoundError(f"Missing metadata YAML: {metadata_yaml}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing encoder file: {encoder_path}")

    # Load metadata & encoder
    md4 = load_planetscope_sr4_metadata(metadata_yaml)
    md8 = load_planetscope_sr8_metadata(metadata_yaml)
    encoder = load_encoder(encoder_path)

    # Construct normalizers once per sensor variant
    norm4 = v2.Compose(
        [
            v2.Normalize(
                mean=[md4["bands"]["mean"][b] for b in md4["band_order"]],
                std=[md4["bands"]["std"][b] for b in md4["band_order"]],
            )
        ]
    )
    norm8 = v2.Compose(
        [
            v2.Normalize(
                mean=[md8["bands"]["mean"][b] for b in md8["band_order"]],
                std=[md8["bands"]["std"][b] for b in md8["band_order"]],
            )
        ]
    )

    roots = [
        Path("/Volumes/x10pro/estuary/low_quality/dove/results"),
        Path("/Volumes/x10pro/estuary/low_quality/superdove/results"),
        Path("/Volumes/x10pro/estuary/ca_all/dove/results"),
        Path("/Volumes/x10pro/estuary/ca_all/superdove/results"),
    ]
    all_paths = discover_sr_paths(roots)
    if not all_paths:
        logger.warning("No SR TIFFs found under expected roots.")
        return

    by_region = group_by_region(all_paths)

    total = sum(len(v) for v in by_region.values())
    pbar = tqdm.tqdm(total=total, desc="Embedding SR tiles")

    # Torch CPU threading can compete with numpy; set to a sensible default
    torch.set_num_threads(max(1, torch.get_num_threads()))

    for region, paths in by_region.items():
        region_save = save_dir / str(region)
        for tif_path in paths:
            key = Path(tif_path).stem
            out = region_save / f"{key}.npy"
            if out.exists() and not overwrite:
                pbar.update(1)
                continue

            # Choose metadata/normalizer based on band count (4 vs 8)
            try:
                with rasterio.open(tif_path) as src:
                    count = src.count
            except Exception as e:
                logger.error(f"Failed to open {tif_path}: {e}")
                pbar.update(1)
                continue

            if count == 4:
                band_meta, norm = md4, norm4
            elif count == 8:
                band_meta, norm = md8, norm8
            else:
                logger.warning(f"Unexpected band count={count} in {tif_path}; skipping.")
                pbar.update(1)
                continue

            emb = encode_one(encoder, band_meta, Path(tif_path), size=size, norm=norm)
            save_embedding(region_save, key, emb, overwrite=overwrite)
            pbar.update(1)

    pbar.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
