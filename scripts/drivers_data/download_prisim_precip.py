import tempfile
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import tqdm
from rasterio.mask import mask

SLEEP_SECONDS = 0.1
REQUEST_TIMEOUT = 60
BASE_URL = "https://services.nacse.org/prism/data/get"
# PRISM web service returns one grid per request. By default the response is a zip
# package, and the server may provide the canonical output filename via
# Content-Disposition.


def prism_zip_url(day: date, resolution: str) -> str:
    ymd = day.strftime("%Y%m%d")
    # PRISM web service uses 4km and 800m in the URL path.
    return f"{BASE_URL}/us/{resolution}/ppt/{ymd}"


def prism_zip_name(day: date, resolution: str) -> str:
    ymd = day.strftime("%Y%m%d")
    return f"prism_ppt_us_{resolution}_{ymd}.zip"


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def download_file(url: str, out_path: Path) -> Path | None:
    r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    final_path = out_path
    content_disposition = r.headers.get("Content-Disposition", "")
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=", 1)[1].strip().strip('"')
        if filename:
            final_path = out_path.parent / filename

    with open(final_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return final_path


def load_watersheds(watersheds_path: Path) -> gpd.GeoDataFrame:
    gdf_poly = gpd.read_file(watersheds_path, layer="GlobalWatershed").to_crs(4326)
    gdf_poly = gdf_poly.rename(columns={"Name": "site_id"})
    gdf_poly = gdf_poly.sort_values("site_id").drop_duplicates(subset="site_id")
    if "site_id" not in gdf_poly.columns:
        raise click.ClickException("Watershed dataset is missing site_id after renaming Name")
    return gdf_poly[["site_id", "geometry"]].copy()


def simplify_watersheds(
    watersheds: gpd.GeoDataFrame, tolerance_m: float = 1000.0
) -> gpd.GeoDataFrame:
    if tolerance_m <= 0:
        return watersheds
    search_crs = "EPSG:3310"
    watersheds_proj = watersheds.to_crs(search_crs).copy()
    watersheds_proj["geometry"] = watersheds_proj.geometry.simplify(
        tolerance=tolerance_m,
        preserve_topology=True,
    )
    return watersheds_proj.to_crs(4326)


def find_raster_file(extract_dir: Path) -> Path:
    tifs = sorted(extract_dir.rglob("*.tif"))
    if tifs:
        return tifs[0]
    bils = sorted(extract_dir.rglob("*.bil"))
    if bils:
        return bils[0]
    raise click.ClickException(f"No .tif or .bil file found under {extract_dir}")


def summarize_one_raster_day(
    raster_path: Path, watersheds: gpd.GeoDataFrame, day: date
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with rasterio.open(raster_path) as src:
        watersheds_src = watersheds.to_crs(src.crs)
        nodata = src.nodata

        for row in watersheds_src.itertuples(index=False):
            geom = row.geometry
            try:
                data, _ = mask(src, [geom], crop=True, filled=False)
            except ValueError:
                rows.append(
                    {
                        "site_id": row.site_id,
                        "date": pd.Timestamp(day),
                        "ppt_mm_mean": np.nan,
                        "ppt_volume_m3": np.nan,
                        "n_pixels": 0,
                    }
                )
                continue

            arr = data[0]
            if np.ma.isMaskedArray(arr):
                vals = arr.compressed()
            else:
                vals = arr.ravel()
                if nodata is not None:
                    vals = vals[vals != nodata]
                vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                ppt_mm_mean = np.nan
                ppt_volume_m3 = np.nan
                n_pixels = 0
            else:
                vals = vals.astype(float)
                vals = vals[np.isfinite(vals)]
                ppt_mm_mean = float(vals.mean()) if vals.size else np.nan
                ppt_volume_m3 = (
                    float((ppt_mm_mean / 1000.0) * row.area_m2)  # type: ignore
                    if vals.size and pd.notna(row.area_m2)
                    else np.nan
                )
                n_pixels = int(vals.size)

            rows.append(
                {
                    "site_id": row.site_id,
                    "date": pd.Timestamp(day),
                    "ppt_mm_mean": ppt_mm_mean,
                    "ppt_volume_m3": ppt_volume_m3,
                    "n_pixels": n_pixels,
                }
            )

    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--start", required=True, type=str, help="YYYY-MM-DD", default="2017-01-01", show_default=True
)
@click.option(
    "--end", required=True, type=str, help="YYYY-MM-DD", default="2025-01-01", show_default=True
)
@click.option("--resolution", type=click.Choice(["4km", "800m"]), default="4km", show_default=True)
@click.option(
    "--watersheds-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to StreamStats geodatabase containing GlobalWatershed layer.",
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to output parquet file containing daily watershed precipitation summaries.",
)
def main(
    start: str,
    end: str,
    resolution: str,
    watersheds_path: Path,
    out_path: Path,
) -> None:
    start_day = date.fromisoformat(start)
    end_day = date.fromisoformat(end)
    watersheds = load_watersheds(watersheds_path)
    watersheds = simplify_watersheds(watersheds, tolerance_m=1000.0)

    watersheds_area = watersheds.to_crs("EPSG:3310")
    watersheds["area_m2"] = watersheds_area.geometry.area.values

    all_days: list[pd.DataFrame] = []
    failures: list[str] = []

    for day in tqdm.tqdm(
        list(daterange(start_day, end_day)), desc="Downloading and summarizing PRISM daily ppt"
    ):
        url = prism_zip_url(day, resolution)
        zip_name = prism_zip_name(day, resolution)

        with tempfile.TemporaryDirectory(prefix="prism_day_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            zip_path = tmpdir_path / zip_name
            extract_dir = tmpdir_path / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)

            downloaded_path = download_file(url, zip_path)
            if downloaded_path is None:
                failures.append(day.isoformat())
                continue

            if downloaded_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(downloaded_path, "r") as zf:
                    zf.extractall(extract_dir)
                raster_path = find_raster_file(extract_dir)
            else:
                raster_path = downloaded_path

            day_df = summarize_one_raster_day(raster_path, watersheds=watersheds, day=day)
            all_days.append(day_df)

        time.sleep(SLEEP_SECONDS)

    if not all_days:
        raise click.ClickException("No PRISM days were successfully processed")

    out_df = pd.concat(all_days, ignore_index=True)
    out_df = out_df.sort_values(["site_id", "date"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    click.echo(f"Wrote {len(out_df)} summarized rows to {out_path}")
    if failures:
        click.echo(f"Missing or failed days: {len(failures)}")
        for d in failures[:20]:
            click.echo(f"  {d}")
        if len(failures) > 20:
            click.echo("  ...")


if __name__ == "__main__":
    main()
