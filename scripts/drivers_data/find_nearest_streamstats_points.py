#!/usr/bin/env python3
import zipfile
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tqdm
from pyproj import Geod, Transformer
from rasterio.transform import xy
from rasterio.windows import Window

# lat, lon
MANUAL_OVERRIDE = {
    19: (34.13601977297579, -119.18169819648033),
    20: (34.23519221944245, -119.25551221585381),
    27: (34.412536075331275, -119.68974425310685),
    28: (34.41801329414458, -119.8296285105292),
    32: (34.79809070036205, -120.62063194463595),
    34: (35.100876648999446, -120.63034167466094),
    35: (35.134908763238926, -120.63979867460078),
    39: (35.59513133841304, -121.12675892870955),
    43: (35.70957863936823, -121.30757533546424),
    46: (36.33488211347528, -121.89201424343617),
    51: (36.852510557100324, -121.8077958351768),
    56: (36.99155995921911, -122.03225529751451),
    57: (36.97235746632335, -121.95297542630927),
    63: (37.09641591959323, -122.27794807437756),
    67: (38.267604428860274, -122.96393660406646),
    72: (38.448964553099955, -123.12399273305685),
    79: (38.945492451651184, -123.72545596756548),
    81: (39.00443520991724, -123.69637383420466),
    84: (39.19295917341879, -123.7602026459383),
    86: (39.471446932667796, -123.80379068424458),
    94: (41.17126688542579, -124.11998103564432),
    95: (41.245347857237924, -124.10210911898433),
    2138: (37.26429153332324, -122.40618735213741),
    2147: (37.83081635560188, -122.5323569053046),
    12097: (41.40347257130567, -124.0680708589927),
    12103: (41.02740020023125, -124.10961556631196),
    13008: (39.539350748305516, -123.750142532374),
    13027: (38.99066798961385, -123.70156439278513),
    13057: (37.32161865041921, -122.40254068107868),
    13099: (35.18197688707747, -120.72814556899145),
}


def _extract_candidate_points(geometry) -> list[tuple[str, float, float]]:
    """
    Return candidate sampling points from a site geometry as
    (candidate_name, lon, lat) tuples in EPSG:4326.

    Uses centroid plus the four bounding-box corners. This is intended
    for the small square AOIs used to represent estuary mouths.
    """
    minx, miny, maxx, maxy = geometry.bounds
    centroid = geometry.centroid
    return [
        ("centroid", float(centroid.x), float(centroid.y)),
        # ("ll", float(minx), float(miny)),
        # ("ul", float(minx), float(maxy)),
        # ("lr", float(maxx), float(miny)),
        # ("ur", float(maxx), float(maxy)),
    ]


def _load_streamgrid_valid_points(
    streamgrid_path: Path,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    chunk_size: int,
) -> gpd.GeoDataFrame:
    with rasterio.open(streamgrid_path) as src:
        if src.crs is None:
            raise click.ClickException("streamgrid raster has no CRS defined")

        transformer_to_raster = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        bbox_xs, bbox_ys = transformer_to_raster.transform(
            [min_lon, min_lon, max_lon, max_lon],
            [min_lat, max_lat, min_lat, max_lat],
        )

        min_x = min(bbox_xs)
        max_x = max(bbox_xs)
        min_y = min(bbox_ys)
        max_y = max(bbox_ys)

        row_min, col_min = src.index(min_x, max_y)
        row_max, col_max = src.index(max_x, min_y)

        row_start = max(0, min(row_min, row_max))
        row_stop = min(src.height, max(row_min, row_max) + 1)
        col_start = max(0, min(col_min, col_max))
        col_stop = min(src.width, max(col_min, col_max) + 1)

        valid_lon_chunks: list[np.ndarray] = []
        valid_lat_chunks: list[np.ndarray] = []
        valid_idx_chunks: list[np.ndarray] = []
        running_idx = 0

        total_windows = (((row_stop - row_start) + chunk_size - 1) // chunk_size) * (
            ((col_stop - col_start) + chunk_size - 1) // chunk_size
        )

        with tqdm.tqdm(total=total_windows, desc="Reading streamgrid windows") as pbar:
            for r0 in range(row_start, row_stop, chunk_size):
                height = min(chunk_size, row_stop - r0)
                for c0 in range(col_start, col_stop, chunk_size):
                    width = min(chunk_size, col_stop - c0)
                    window = Window(col_off=c0, row_off=r0, width=width, height=height)  # type: ignore
                    arr = src.read(1, window=window, masked=True)

                    if np.ma.isMaskedArray(arr):
                        valid_mask = ~arr.mask
                    else:
                        nodata = src.nodata
                        if nodata is None:
                            valid_mask = np.ones(arr.shape, dtype=bool)
                        else:
                            valid_mask = arr != nodata

                    if np.any(valid_mask):
                        rows, cols = np.where(valid_mask)
                        rows = rows + r0
                        cols = cols + c0
                        xs, ys = xy(src.transform, rows, cols, offset="center")
                        xs = np.asarray(xs, dtype=float)
                        ys = np.asarray(ys, dtype=float)
                        lons, lats = transformer_to_wgs84.transform(xs, ys)
                        lons = np.asarray(lons, dtype=float)
                        lats = np.asarray(lats, dtype=float)

                        in_bbox = (
                            (lons >= min_lon)
                            & (lons <= max_lon)
                            & (lats >= min_lat)
                            & (lats <= max_lat)
                        )
                        if np.any(in_bbox):
                            kept = int(np.count_nonzero(in_bbox))
                            valid_lon_chunks.append(lons[in_bbox])
                            valid_lat_chunks.append(lats[in_bbox])
                            valid_idx_chunks.append(np.arange(running_idx, running_idx + kept))
                            running_idx += kept

                    pbar.update(1)

    if not valid_lon_chunks:
        return gpd.GeoDataFrame(
            {"streamgrid_idx": pd.Series(dtype=int)},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    valid_lon = np.concatenate(valid_lon_chunks)
    valid_lat = np.concatenate(valid_lat_chunks)
    valid_idx = np.concatenate(valid_idx_chunks)

    return gpd.GeoDataFrame(
        {"streamgrid_idx": valid_idx, "pt_lon": valid_lon, "pt_lat": valid_lat},
        geometry=gpd.points_from_xy(valid_lon, valid_lat),
        crs="EPSG:4326",
    )


@click.command()
@click.option(
    "--streamgrid-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to StreamStats streamgrid.tif raster.",
)
@click.option(
    "--grid-points-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help=(
        "Path to a directory of per-site geospatial files readable by GeoPandas. "
        "Each filename stem is used as the site identifier."
    ),
)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output file. Use .parquet, .csv, .geojson, .gpkg, or .shp",
)
@click.option(
    "--buffer-deg",
    default=1.0,
    show_default=True,
    type=float,
    help="Latitude/longitude buffer, in degrees, for subsetting candidate streamgrid cells.",
)
@click.option(
    "--chunk-size",
    default=2048,
    show_default=True,
    type=int,
    help="Raster window size in pixels for chunked streamgrid reads.",
)
def main(
    streamgrid_path: Path,
    grid_points_dir: Path,
    out_path: Path,
    buffer_deg: float,
    chunk_size: int,
) -> None:
    """
    Find the nearest valid StreamStats streamgrid point for each site geometry.

    Output columns:
    - site_id
    - mouth_lon, mouth_lon_360, mouth_lat
    - pt_lon, pt_lat
    - candidate_name
    - dist_m, dist_km
    """
    click.echo(f"Reading site geometries from: {grid_points_dir}")
    site_files = sorted(
        [p for p in grid_points_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    )
    if not site_files:
        raise click.ClickException(f"No site files found in: {grid_points_dir}")

    site_candidates: list[dict[str, object]] = []
    for site_file in site_files:
        site_id = site_file.stem
        gdf = gpd.read_file(site_file)

        if gdf.empty:
            click.echo(f"Skipping empty site file: {site_file.name}")
            continue

        if gdf.crs is None:
            click.echo(f"Warning: {site_file.name} has no CRS defined. Assuming EPSG:4326.")
            gdf = gdf.set_crs(4326)
        elif str(gdf.crs).lower() != "epsg:4326":
            click.echo(f"Reprojecting {site_file.name} from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(4326)

        geom = gdf.geometry.iloc[0]
        site_id_int = int(site_id)
        for candidate_name, lon_raw, lat_raw in _extract_candidate_points(geom):
            # manual overrides
            if site_id_int in MANUAL_OVERRIDE:
                lat_raw, lon_raw = MANUAL_OVERRIDE[site_id_int]

            site_candidates.append(
                {
                    "site_id": site_id,
                    "candidate_name": candidate_name,
                    "candidate_lon": lon_raw,
                    "candidate_lat": lat_raw,
                    "candidate_lon_360": (lon_raw + 360) % 360,
                }
            )

    if not site_candidates:
        raise click.ClickException("No valid site geometries were found.")

    candidates_df = pd.DataFrame(site_candidates)

    min_lon = candidates_df["candidate_lon"].min() - buffer_deg
    max_lon = candidates_df["candidate_lon"].max() + buffer_deg
    min_lat = candidates_df["candidate_lat"].min() - buffer_deg
    max_lat = candidates_df["candidate_lat"].max() + buffer_deg

    if chunk_size < 1:
        raise click.ClickException("chunk-size must be >= 1")

    click.echo(f"Loading valid streamgrid cells from: {streamgrid_path}")
    valid_gdf = _load_streamgrid_valid_points(
        streamgrid_path=streamgrid_path,
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        chunk_size=chunk_size,
    )
    click.echo(f"Loaded {len(valid_gdf)} valid streamgrid cells inside search bounds")

    if valid_gdf.empty:
        raise click.ClickException("No valid streamgrid cells found inside the search window.")

    candidate_gdf = gpd.GeoDataFrame(
        candidates_df.copy(),
        geometry=gpd.points_from_xy(candidates_df["candidate_lon"], candidates_df["candidate_lat"]),
        crs="EPSG:4326",
    )

    search_crs = "EPSG:3310"
    candidate_proj = candidate_gdf.to_crs(search_crs)
    valid_proj = valid_gdf.to_crs(search_crs)

    click.echo("Running nearest-neighbor join in projected space")
    joined = gpd.sjoin_nearest(
        candidate_proj,
        valid_proj[["streamgrid_idx", "pt_lon", "pt_lat", "geometry"]],
        how="left",
        distance_col="dist_join_m",
    )

    if joined["streamgrid_idx"].isna().any():
        raise click.ClickException(
            "Failed to match one or more candidate points to streamgrid cells"
        )

    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(
        joined["candidate_lon"].to_numpy(),
        joined["candidate_lat"].to_numpy(),
        joined["pt_lon"].to_numpy(),
        joined["pt_lat"].to_numpy(),
    )

    candidate_df = pd.DataFrame(
        {
            "site_id": joined["site_id"].to_numpy(),
            "candidate_name": joined["candidate_name"].to_numpy(),
            "mouth_lon": joined["candidate_lon"].to_numpy(),
            "mouth_lat": joined["candidate_lat"].to_numpy(),
            "mouth_lon_360": joined["candidate_lon_360"].to_numpy(),
            "pt_lon": joined["pt_lon"].to_numpy(),
            "pt_lat": joined["pt_lat"].to_numpy(),
            "pt_lon_360": (joined["pt_lon"].to_numpy() + 360) % 360,
            "streamgrid_idx": joined["streamgrid_idx"].astype(int).to_numpy(),
            "dist_m": dist_m,
            "dist_km": dist_m / 1000.0,
        }
    )

    out_df = (
        candidate_df.sort_values(["site_id", "dist_m", "candidate_name"])
        .groupby("site_id", as_index=False)
        .first()
        .sort_values("site_id")
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    click.echo(f"Writing output to: {out_path}")

    if suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        out_df.to_csv(out_path, index=False)
    elif suffix in {".geojson", ".gpkg"}:
        out_gdf = gpd.GeoDataFrame(
            out_df,
            geometry=gpd.points_from_xy(out_df["pt_lon"], out_df["pt_lat"]),
            crs="EPSG:4326",
        )
        driver = "GeoJSON" if suffix == ".geojson" else "GPKG"
        out_gdf.to_file(out_path, driver=driver)

    elif suffix == ".shp":
        # write shapefile in streamgrid CRS
        batch_df = out_df.copy()
        batch_df.insert(0, "ID", batch_df["site_id"].astype(str))

        # NOTE: streamgrid CRS assumed same as raster CRS; using EPSG:4326 fallback if needed
        batch_gdf = gpd.GeoDataFrame(
            batch_df,
            geometry=gpd.points_from_xy(batch_df["pt_lon"], batch_df["pt_lat"]),
            crs="EPSG:4326",
        )

        batch_gdf.to_file(out_path)

        # zip shapefile components
        shp_base = out_path.with_suffix("")
        zip_path = out_path.parent.with_suffix(".zip")

        with zipfile.ZipFile(zip_path, "w") as z:
            for ext in [".shp", ".shx", ".dbf", ".prj"]:
                f = shp_base.with_suffix(ext)
                if f.exists():
                    z.write(f, arcname=f.name)

        click.echo(f"Zipped shapefile to {zip_path}")

        if len(batch_gdf) > 250:
            click.echo("Warning: StreamStats Batch Processor only processes the first 250 points.")
    else:
        raise click.ClickException(
            "Unsupported output format. Use .parquet, .csv, .geojson, .gpkg, or .shp"
        )

    click.echo(f"Selected nearest points for {len(out_df)} sites.")
    click.echo("Candidate counts by selected candidate:")
    click.echo(out_df["candidate_name"].value_counts().to_string())

    click.echo("\nSummary:")
    click.echo(out_df["dist_km"].describe().to_string())


if __name__ == "__main__":
    main()
