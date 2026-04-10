#!/usr/bin/env python3

from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from pyproj import Geod


def _to_360(lon: np.ndarray | float) -> np.ndarray | float:
    return (lon + 360) % 360


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
        ("ll", float(minx), float(miny)),
        ("ul", float(minx), float(maxy)),
        ("lr", float(maxx), float(miny)),
        ("ur", float(maxx), float(maxy)),
    ]


@click.command()
@click.option(
    "--tide-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the FES tide directory containing constituent NetCDF files, e.g. ocean_tide_20241025/",
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
    help="Path to output file. Recommended: .parquet, .csv, or .geojson",
)
@click.option(
    "--constituent",
    default="m2_fes2022.nc",
    show_default=True,
    help="Constituent file used only to derive valid model grid cells.",
)
@click.option(
    "--skipped-column",
    default="skipped",
    show_default=True,
    help="Boolean column used to filter out skipped sites. Ignored if not present.",
)
@click.option(
    "--buffer-deg",
    default=1.0,
    show_default=True,
    type=float,
    help="Latitude/longitude buffer, in degrees, for subsetting candidate tide grid cells.",
)
def main(
    tide_dir: Path,
    grid_points_dir: Path,
    out_path: Path,
    constituent: str,
    skipped_column: str,
    buffer_deg: float,
) -> None:
    """
    Find the nearest valid FES model point for each site geometry in a directory.

    Output columns:
    - site_id
    - mouth_lon_360, mouth_lon, mouth_lat
    - pt_lon_360, pt_lon, pt_lat
    - dist_m, dist_km
    """
    constituent_path = tide_dir / constituent
    if not constituent_path.exists():
        raise click.ClickException(f"Could not find constituent file: {constituent_path}")

    click.echo(f"Opening tide grid from: {constituent_path}")
    ds = xr.open_dataset(constituent_path)

    # FES2022 ocean tide files typically expose these variables
    if "lon" not in ds or "lat" not in ds:
        raise click.ClickException("Expected dataset variables 'lon' and 'lat' were not found.")
    if "amplitude" not in ds:
        raise click.ClickException("Expected dataset variable 'amplitude' was not found.")

    lon = ds["lon"].values
    lat = ds["lat"].values
    amp = ds["amplitude"].values

    valid = np.isfinite(amp)

    if lon.ndim == 1 and lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lon2d, lat2d = lon, lat

    valid_lon = lon2d[valid]
    valid_lat = lat2d[valid]

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
        for candidate_name, lon_raw, lat_raw in _extract_candidate_points(geom):
            site_candidates.append(
                {
                    "site_id": site_id,
                    "candidate_name": candidate_name,
                    "candidate_lon": lon_raw,
                    "candidate_lat": lat_raw,
                    "candidate_lon_360": _to_360(lon_raw),
                }
            )

    if not site_candidates:
        raise click.ClickException("No valid site geometries were found.")

    candidates_df = pd.DataFrame(site_candidates)

    min_lon = _to_360(candidates_df["candidate_lon"].min() - buffer_deg)
    max_lon = _to_360(candidates_df["candidate_lon"].max() + buffer_deg)
    min_lat = candidates_df["candidate_lat"].min() - buffer_deg
    max_lat = candidates_df["candidate_lat"].max() + buffer_deg

    # This simple filter assumes the longitude window does not wrap around 0.
    # That is true for California in [0, 360] space.
    mask = (
        (valid_lon > min_lon)
        & (valid_lon < max_lon)
        & (valid_lat > min_lat)
        & (valid_lat < max_lat)
    )
    filt_valid_lon = valid_lon[mask]
    filt_valid_lat = valid_lat[mask]

    if len(filt_valid_lon) == 0:
        raise click.ClickException("No valid tide grid cells found inside the search window.")

    geod = Geod(ellps="WGS84")

    def nearest_valid_model_point(
        target_lon_360: float,
        target_lat: float,
        search_lon_360: np.ndarray,
        search_lat: np.ndarray,
    ) -> tuple[float, float, float]:
        _, _, dist_m = geod.inv(
            np.full(search_lon_360.shape, target_lon_360),
            np.full(search_lat.shape, target_lat),
            search_lon_360,
            search_lat,
        )
        i = int(np.argmin(dist_m))
        return float(search_lon_360[i]), float(search_lat[i]), float(dist_m[i])

    candidate_rows: list[dict[str, object]] = []
    candidate_iter = candidates_df.itertuples(index=False)
    for row in tqdm.tqdm(
        candidate_iter,
        total=len(candidates_df),
        desc="Finding nearest valid tide points",
    ):
        pt_lon_360, pt_lat, dist_m = nearest_valid_model_point(
            target_lon_360=row.candidate_lon_360,  # pyright: ignore[reportArgumentType]
            target_lat=row.candidate_lat,  # pyright: ignore[reportArgumentType]
            search_lon_360=filt_valid_lon,
            search_lat=filt_valid_lat,
        )
        pt_lon = ((pt_lon_360 + 180) % 360) - 180

        candidate_rows.append(
            {
                "site_id": row.site_id,
                "candidate_name": row.candidate_name,
                "mouth_lon_360": row.candidate_lon_360,
                "mouth_lon": row.candidate_lon,
                "mouth_lat": row.candidate_lat,
                "pt_lon_360": pt_lon_360,
                "pt_lon": pt_lon,
                "pt_lat": pt_lat,
                "dist_m": dist_m,
                "dist_km": dist_m / 1000.0,
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
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
    elif suffix in {".geojson", ".gpkg", ".shp"}:
        out_gdf = gpd.GeoDataFrame(
            out_df,
            geometry=gpd.points_from_xy(out_df["pt_lon"], out_df["pt_lat"]),
            crs="EPSG:4326",
        )
        driver = None
        if suffix == ".geojson":
            driver = "GeoJSON"
        elif suffix == ".gpkg":
            driver = "GPKG"
        out_gdf.to_file(out_path, driver=driver)
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
