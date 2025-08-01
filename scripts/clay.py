import datetime
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import torch
import tqdm
import yaml
from pyproj import Transformer
from torchvision.transforms import v2

# Load sensor metadata
with open("/Users/kyledorman/data/models/clay/metadata.yaml") as f:
    metadata = yaml.safe_load(f)

channel_4_band_order = [
    "blue",
    "green",
    "red",
    "nir",
]
planetscope = metadata["planetscope-sr"]
metadata["planetscope-sr-4"] = {}
metadata["planetscope-sr-4"]["band_order"] = channel_4_band_order
metadata["planetscope-sr-4"]["rgb_indices"] = [3, 2, 1]
metadata["planetscope-sr-4"]["gsd"] = 3
bands = {}
for k, vs in planetscope["bands"].items():
    vs4 = {kk: vv for kk, vv in vs.items() if kk in channel_4_band_order}
    bands[k] = vs4
metadata["planetscope-sr-4"]["bands"] = bands

metadata["planetscope-sr-4"]


def parse_dt_from_pth(pth: str):
    datetime_str = "_".join(Path(pth).stem.split("_")[:2])
    date_format = "%Y%m%d_%H%M%S"
    dt = datetime.datetime.strptime(datetime_str, date_format)
    return dt


def load_tif(
    pth: Path, crop: tuple[int, int, int, int], out_size: int
) -> tuple[np.ndarray, tuple[float, float], float]:
    with rasterio.open(pth) as src:
        bounds = src.bounds
        crs = src.crs

        transformer = Transformer.from_crs(crs, 4326)
        # Calculate centroid
        centroid_x = (bounds.left + bounds.right) / 2
        centroid_y = (bounds.top + bounds.bottom) / 2
        centroid_x, centroid_y = transformer.transform(centroid_x, centroid_y)
        cent_g = centroid_x, centroid_y

        # plt.figure()
        # show((src, 1), cmap='viridis')
        assert src.count == 4
        data = src.read()
        start_w, start_h, end_w, end_h = crop
        w = end_w - start_w
        gsd = src.meta["transform"][0]
        gsd = gsd * w / out_size
        data = data[:, start_h:end_h, start_w:end_w]

        return data, cent_g, gsd


def normalize_latlon(lat, lon):
    """
    Normalize latitude and longitude to a range between -1 and 1.

    Parameters:
    lat (float): Latitude value.
    lon (float): Longitude value.

    Returns:
    tuple: Normalized latitude and longitude values.
    """
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def prep_datacube(
    image: np.ndarray, lat: float, lon: float, date: datetime.datetime, gsd: float, size: int
):
    """
    Prepare a data cube for model input.

    Parameters:
    image (np.array): The input image array.
    lat (float): Latitude value for the location.
    lon (float): Longitude value for the location.
    date (datetime): capture datetime
    gsd: (float): the scaled ground sampling distance

    Returns:
    dict: Prepared data cube with normalized values and embeddings.
    """
    md = metadata["planetscope-sr-4"]

    # Extract mean, std, and wavelengths from metadata
    mean = []
    std = []
    waves = []
    bands = md["band_order"]
    for band_name in bands:
        mean.append(md["bands"]["mean"][band_name])
        std.append(md["bands"]["std"][band_name])
        waves.append(md["bands"]["wavelength"][band_name] * 1000)

    transform = v2.Compose(
        [
            v2.Resize(size=(size, size), interpolation=3),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    # Prep datetimes embedding
    times = normalize_timestamp(date)
    week_norm = times[0]
    hour_norm = times[1]

    # Prep lat/lon embedding
    latlons = normalize_latlon(lat, lon)
    lat_norm = latlons[0]
    lon_norm = latlons[1]

    # Prep pixels
    pixels = torch.from_numpy(image.astype(np.float32))
    pixels = transform(pixels)
    pixels = pixels.unsqueeze(0)

    # Prepare additional information
    return {
        "pixels": pixels,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
        ).unsqueeze(0),
        "latlon": torch.tensor(np.hstack((lat_norm, lon_norm)), dtype=torch.float32).unsqueeze(0),
        "waves": torch.tensor(waves),
        "gsd": torch.tensor(gsd).unsqueeze(0),
    }


ep_embedder_cpu = torch.export.load(
    "/Users/kyledorman/data/models/clay/clay-v1.5-encoder_256.pt2"
).module()

all_paths = {}
dove = Path("/Users/kyledorman/data/estuary/dove/results")
SKIP_REGION = ["arroyo_sequit", "drakes_estero", "eel_river", "mugu_lagoon", "batiquitos_lagoon"]
SIZE = 256

regions = sorted(
    list({p.stem for p in dove.glob("*/*/*") if p.stem not in SKIP_REGION and p.is_dir()})
)
for region in regions:
    all_tifs = list(dove.glob(f"*/*/{region}/files/*_SR_clip.tif"))
    for pth in all_tifs:
        key = pth.stem.replace("_SR_clip", "")
        all_paths[key] = pth

base = Path("/Users/kyledorman/data/estuary/label_studio/00025")
with open(base.parent / "region_crops.json") as f:
    region_crops = json.load(f)  # noqa: F821

to_run = list(base.glob("*/images/*.jpg"))
for pth in tqdm.tqdm(to_run):
    key = pth.stem
    tpth = all_paths[key]

    save_dir = pth.parent.parent / "embeddings"
    save_dir.mkdir(exist_ok=True, parents=True)
    sv_pth = save_dir / f"{key}.npy"
    if sv_pth.exists():
        continue

    data, cent_g, gsd = load_tif(tpth, region_crops[pth.parent.parent.name], SIZE)
    dt = parse_dt_from_pth(tpth)

    datacube = prep_datacube(data, *cent_g, dt, gsd, SIZE)
    with torch.no_grad():
        embeddings = ep_embedder_cpu(datacube).detach().cpu().numpy()[0]

    np.save(sv_pth, embeddings)
