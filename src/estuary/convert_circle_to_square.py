import math
import warnings
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon, box
from tqdm import tqdm

# If your radius is provided as a property on a Point, list candidate keys
# and units (assumed meters if working in projected CRS; for EPSG:4326, meters):
POINT_RADIUS_KEYS = ["radius_m", "radius", "r_m"]


INPUT_DIR = Path("/Volumes/x10pro/estuary/recropped")
OUTPUT_DIR = Path("/Volumes/x10pro/estuary/recropped_rect")

# ----------------- Helpers ---------------------------------------------------


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """Return WGS84 UTM EPSG code for coordinate (lon, lat)."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def ensure_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """If the GeoJSON has no CRS, assume WGS84 (EPSG:4326)."""
    if gdf.crs is None:
        warnings.warn("Input file has no CRS; assuming EPSG:4326 (lon/lat).", stacklevel=2)
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


def circle_center_radius_m(gdf_single: gpd.GeoDataFrame) -> tuple[float, float, float, CRS, CRS]:
    """
    Infer circle center (cx, cy) and radius (meters).
    Accepts either:
      - Polygon approximating a circle
      - Point with a numeric radius property (meters)
    Returns (cx, cy) in local projected CRS, radius_m (float), original CRS, local CRS
    """
    assert len(gdf_single) == 1, "Expected one feature per file"
    gdf = ensure_crs(gdf_single)
    geom = gdf.geometry.iloc[0]
    orig_crs = gdf.crs
    assert orig_crs is not None

    assert isinstance(geom, Polygon)

    # Otherwise treat geometry as a polygon approximating a circle.
    # Project to local UTM for metric distance.
    if CRS.from_user_input(orig_crs).is_geographic:
        lon, lat = geom.centroid.x, geom.centroid.y
        utm = CRS.from_epsg(utm_epsg_from_lonlat(lon, lat))
        gdf_loc = gdf.to_crs(utm)
    else:
        utm = CRS.from_user_input(orig_crs)
        gdf_loc = gdf

    poly = gdf_loc.geometry.iloc[0]
    assert isinstance(poly, Polygon)

    # Estimate radius as max distance from centroid to polygon boundary points
    c = poly.centroid
    cx, cy = c.x, c.y
    xs, ys = poly.exterior.xy
    # max Euclidean distance to exterior vertices
    r = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys, strict=False))

    return cx, cy, float(r), orig_crs, utm


def circumscribed_square(cx: float, cy: float, r: float) -> Polygon:
    """
    Build axis-aligned square that circumscribes a circle of radius r centered at (cx, cy).
    Side length = 2r. Returns a shapely Polygon in same (projected) CRS.
    """
    return box(cx - r, cy - r, cx + r, cy + r)


inputs = sorted(list(INPUT_DIR.glob("*.geojson")))

if not inputs:
    print(f"No GeoJSON files found in {INPUT_DIR}")

for src in tqdm(inputs, desc="Circles -> Squares"):
    try:
        gdf = gpd.read_file(src)
        if len(gdf) != 1:
            raise ValueError(f"{src.name}: expected exactly 1 feature, found {len(gdf)}")

        cx, cy, r_m, orig_crs, metr_crs = circle_center_radius_m(gdf)
        square_m = gpd.GeoDataFrame(
            gdf.drop(columns="geometry"), geometry=[circumscribed_square(cx, cy, r_m)], crs=metr_crs
        ).to_crs(orig_crs)

        # Write with consistent filename
        dst = OUTPUT_DIR / src.name
        OUTPUT_DIR.mkdir(exist_ok=True)
        square_m.to_file(dst, driver="GeoJSON")
    except Exception as e:
        print(f"[WARN] {src.name}: {e}")

print("Done.")
