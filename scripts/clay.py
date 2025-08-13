"""Embed PlanetScope SR crops into CLAY encoder embeddings.

Refactor highlights:
- Click CLI with required inputs and sensible defaults
- Minimal INFO logging
- TQDM progress bar
- Small, testable functions
- Fail-fast error handling

Usage example:

python -m estuary.scripts.clay \
  --ls-base /Users/kyledorman/data/estuary/label_studio/00025 \
  --region-crops /Users/kyledorman/data/estuary/label_studio/region_crops.json \
  --size 256 \
  --overwrite

"""

import datetime as _dt
import json
import logging
import math
from collections.abc import Iterable
from pathlib import Path

import click
import numpy as np
import rasterio
import torch
import tqdm
import yaml
from pyproj import Transformer
from torchvision.transforms import v2

# -----------------------------
# Constants / defaults
# -----------------------------
DEFAULT_METADATA_YAML = "/Users/kyledorman/data/models/clay/metadata.yaml"
DEFAULT_ENCODER_PATH = "/Users/kyledorman/data/models/clay/clay-v1.5-encoder_256.pt2"
DEFAULT_DOVE_ROOT = "/Users/kyledorman/data/estuary/dove/results"
SKIP_REGION = {"arroyo_sequit", "drakes_estero", "eel_river", "mugu_lagoon", "batiquitos_lagoon"}

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


def parse_dt_from_pth(pth: Path) -> _dt.datetime:
    """Parse acquisition datetime from file stem prefix YYYYMMDD_HHMMSS_*"""
    datetime_str = "_".join(pth.stem.split("_")[:2])
    date_format = "%Y%m%d_%H%M%S"
    return _dt.datetime.strptime(datetime_str, date_format)


def load_tif_crop(
    pth: Path, crop: tuple[int, int, int, int], out_size: int
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Read 4-band SR GeoTIFF, crop to window, return array, centroid (lat, lon), and scaled GSD.

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

        start_w, start_h, end_w, end_h = crop
        w = end_w - start_w
        gsd = src.meta["transform"][0]
        gsd = gsd * w / out_size  # scaled GSD for the resized crop

        data = src.read()[:, start_h:end_h, start_w:end_w]
        return data, (lat, lon), float(gsd)


def normalize_latlon(lat: float, lon: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Encode latitude/longitude onto the unit circle for model input."""
    lat_r = lat * math.pi / 180
    lon_r = lon * math.pi / 180
    return (math.sin(lat_r), math.cos(lat_r)), (math.sin(lon_r), math.cos(lon_r))


def normalize_timestamp(date: _dt.datetime) -> tuple[tuple[float, float], tuple[float, float]]:
    """Encode ISO week-of-year and hour-of-day as sin/cos pairs."""
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = date.hour * 2 * math.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def prep_datacube(
    md4: dict,
    image: np.ndarray,
    lat: float,
    lon: float,
    date: _dt.datetime,
    gsd: float,
    size: int,
):
    """Prepare a dict of tensors for the encoder forward pass."""
    # Extract channel stats in band order
    mean, std, waves = [], [], []
    for band_name in md4["band_order"]:
        mean.append(md4["bands"]["mean"][band_name])
        std.append(md4["bands"]["std"][band_name])
        waves.append(md4["bands"]["wavelength"][band_name] * 1000)

    transform = v2.Compose(
        [
            v2.Resize(size=(size, size), interpolation=3),  # keep numeric to match existing model
            v2.Normalize(mean=mean, std=std),
        ]
    )

    (week_sin, week_cos), (hour_sin, hour_cos) = normalize_timestamp(date)
    (lat_sin, lat_cos), (lon_sin, lon_cos) = normalize_latlon(lat, lon)

    pixels = torch.from_numpy(image.astype(np.float32))
    pixels = transform(pixels).unsqueeze(0)

    return {
        "pixels": pixels,
        "time": torch.tensor(
            [week_sin, week_cos, hour_sin, hour_cos], dtype=torch.float32
        ).unsqueeze(0),
        "latlon": torch.tensor([lat_sin, lat_cos, lon_sin, lon_cos], dtype=torch.float32).unsqueeze(
            0
        ),
        "waves": torch.tensor(waves),
        "gsd": torch.tensor(gsd).unsqueeze(0),
    }


def discover_sr_paths(dove_root: Path, regions: Iterable[str]) -> dict[str, Path]:
    """Map scene key → SR TIF path by scanning the Dove results tree for each region."""
    all_paths: dict[str, Path] = {}
    for region in regions:
        tifs = dove_root.glob(f"*/*/{region}/files/*_SR_clip.tif")
        for pth in tifs:
            key = pth.stem.replace("_SR_clip", "")
            all_paths[key] = pth
    return all_paths


def region_list_from_ls(ls_base: Path) -> list[str]:
    """List regions under a Label Studio export tree, minus SKIP_REGION."""
    regions = sorted({p.stem for p in ls_base.glob("*/*/*") if p.is_dir()})
    return [r for r in regions if r not in SKIP_REGION]


def iter_label_studio_jpegs(ls_base: Path) -> Iterable[Path]:
    """Yield all Label Studio JPEGs under <region>/images/*.jpg"""
    yield from ls_base.glob("*/images/*.jpg")


def load_encoder(encoder_path: Path):
    """Load the exported Torch encoder (CPU only)."""
    mod = torch.export.load(str(encoder_path)).module()
    mod.eval()
    return mod


def encode_one(
    encoder,
    md4: dict,
    tif_path: Path,
    crop_box: tuple[int, int, int, int],
    size: int,
) -> np.ndarray:
    data, (lat, lon), gsd = load_tif_crop(tif_path, crop_box, size)
    dt = parse_dt_from_pth(tif_path)
    datacube = prep_datacube(md4, data, lat, lon, dt, gsd, size)
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
    "--ls-base",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Label Studio base directory (contains <region>/images/*.jpg).",
)
@click.option(
    "--region-crops",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="JSON mapping of region name -> [start_w, start_h, end_w, end_h].",
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
    ls_base: Path,
    region_crops: Path,
    size: int,
    overwrite: bool,
):
    """Embed all Label Studio image regions using the CLAY encoder (CPU)."""
    _setup_logging()

    # Fail fast on required resources
    if not metadata_yaml.exists():
        raise FileNotFoundError(f"Missing metadata YAML: {metadata_yaml}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing encoder file: {encoder_path}")

    # Load metadata & encoder
    md4 = load_planetscope_sr4_metadata(metadata_yaml)
    encoder = load_encoder(encoder_path)

    # Regions and crops
    regions = region_list_from_ls(ls_base)
    if not regions:
        raise RuntimeError(f"No regions found under {ls_base}")

    with open(region_crops) as f:
        crops_map = json.load(f)

    # Discover SR TIF paths per region
    dove_root = Path(DEFAULT_DOVE_ROOT)
    all_paths = discover_sr_paths(dove_root, regions)
    if not all_paths:
        raise RuntimeError(f"No *_SR_clip.tif paths discovered under {dove_root}")

    # Iterate Label Studio JPEGs
    jpegs = list(iter_label_studio_jpegs(ls_base))
    if not jpegs:
        raise RuntimeError(f"No JPEGs found under {ls_base}/<region>/images/*.jpg")

    logger.info("Encoding %d images (size=%d, overwrite=%s)…", len(jpegs), size, overwrite)
    pbar = tqdm.tqdm(jpegs, unit="img")

    for jpg in pbar:
        # Derive key and region
        key = jpg.stem
        region = jpg.parent.parent.name

        # Resolve TIF path and crop box (fail fast if missing)
        tif_path = all_paths.get(key)
        if tif_path is None:
            raise FileNotFoundError(f"Missing SR TIF for key={key} region={region}")

        if region not in crops_map:
            raise KeyError(f"Missing crop for region={region} in {region_crops}")
        crop_box = tuple(crops_map[region])  # type: ignore[arg-type]
        if len(crop_box) != 4:
            raise ValueError(f"Invalid crop for region={region}: {crop_box}")

        save_dir = jpg.parent.parent / "embeddings"
        out_path = save_embedding(save_dir, key, np.empty(0), overwrite=False)  # probe
        if out_path.exists() and not overwrite:
            # already exists; skip heavy work but advance bar
            pbar.set_description(f"skip {region}/{key}")
            continue

        # Compute embedding
        emb = encode_one(encoder, md4, tif_path, crop_box, size)
        save_embedding(save_dir, key, emb, overwrite=True)
        pbar.set_description(f"ok   {region}/{key}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
