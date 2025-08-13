# save as crops_to_geojson.py
import json
from pathlib import Path

import rasterio
import tqdm
from rasterio.warp import transform_geom
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from shapely.geometry import Polygon, mapping

# ---- INPUTS ----
CROP_PATH = Path("/Users/kyledorman/data/estuary/label_studio/region_crops.json")
REGIONS_PATH = Path("/Users/kyledorman/data/estuary/dove/results")
OUT_DIR = Path("/Users/kyledorman/data/estuary/skysat/tight_grids")
REPROJECT_TO_EPSG4326 = True  # set False to keep each feature in its image CRS


def pixel_box_to_polygon(transform, left, top, right, bottom):
    """
    Given affine transform and pixel bounds (top,row; left,col) etc.,
    make a shapely Polygon in the raster's CRS.
    """
    # Window expects row offsets (y) and col offsets (x)
    height = bottom - top
    width = right - left
    win = Window(col_off=left, row_off=top, width=width, height=height)  # type: ignore

    # Get geospatial bounds for the window (minx, miny, maxx, maxy) in the raster CRS
    minx, miny, maxx, maxy = window_bounds(win, transform=transform)

    # Polygon in that CRS (counter-clockwise, closed ring)
    poly = Polygon([(minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
    return poly


features = []

with open(CROP_PATH) as f:
    data = json.load(f)
    for region, crop in tqdm.tqdm(data.items(), total=len(data)):
        tif_path = list(REGIONS_PATH.glob(f"*/*/{region}/files/*_SR_clip.tif"))[0]

        # Open the raster to grab transform & crs
        with rasterio.open(tif_path) as ds:
            transform = ds.transform
            crs = ds.crs  # may be None if the file lacks CRS (you should fix that upstream)

            # Build polygon in raster CRS
            poly = pixel_box_to_polygon(transform, *crop)
            geom = mapping(poly)

            # Optionally reproject to EPSG:4326 (lon/lat), which is standard for GeoJSON
            if REPROJECT_TO_EPSG4326 and crs is not None:
                geom = transform_geom(crs.to_string(), "EPSG:4326", geom, precision=9)

            # Pack properties you care about
            props = {
                "tif_path": str(tif_path),
                "region": region,
                "source_crs": crs.to_string() if crs else None,
            }

            features = [
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props,
                }
            ]

            fc = {
                "type": "FeatureCollection",
                "name": region,
                "features": features,
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            }
            OUT_DIR.mkdir(exist_ok=True, parents=True)
            with open(OUT_DIR / f"{region}.geojson", "w") as f:
                json.dump(fc, f)
