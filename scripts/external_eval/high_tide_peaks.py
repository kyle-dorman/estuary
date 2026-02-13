"""Compute high-tide peak timestamps for many sites.

This script:
- Loads a GeoJSON/GeoPackage/etc. of site locations.
- Computes a tide-height time series for each site using pyTMD via `calc_tide_elevatons`.
- Finds local maxima (high tides) in each site's time series using `scipy.signal.find_peaks`.
- Writes a JSON mapping of site_id -> list[ISO timestamp].

Notes
-----
- Peaks are detected on the modeled/observed tide series; choose `--distance-samples` to control
  how close peaks can be in *samples*. For 10-minute sampling:
    * ~36 samples ≈ 6 hours (semi-diurnal high tides)
    * ~72 samples ≈ 12 hours (diurnal)
    * ~1008 samples ≈ 7 days (roughly weekly maxima)
- If you want "every high tide", start with ~36.
"""

import datetime
import json
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
from scipy.signal import find_peaks

from estuary.util.tides import calc_tide_elevatons


@click.command(context_settings={"show_default": True})
@click.option(
    "--sites-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input vector file with site locations (GeoJSON/GPKG/etc.).",
)
@click.option(
    "--out-json",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output JSON path (site_id -> list of ISO timestamps).",
)
@click.option(
    "--out-npz",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional output .npz file to save full tide elevation array and time grid.",
)
@click.option(
    "--tide-model-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/Users/kyledorman/data/tides"),
    help="Directory containing pyTMD tide model files.",
)
@click.option(
    "--model-name",
    type=str,
    default="GOT4.10",
    help="pyTMD model name (e.g., GOT4.10).",
)
@click.option(
    "--model-format",
    type=str,
    default="GOT",
    help="pyTMD model format string (e.g., GOT).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]),
    required=True,
    help="Start time (UTC). Common format: YYYY-MM-DD",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]),
    required=True,
    help="End time (UTC). Common format: YYYY-MM-DD",
)
@click.option(
    "--step-minutes",
    type=int,
    default=30,
    help="Sampling cadence in minutes for the tide time series.",
)
@click.option(
    "--distance-samples",
    type=int,
    default=1008,
    help="Minimum peak separation in samples (see module docstring for guidance).",
)
def main(
    sites_path: Path,
    out_json: Path,
    out_npz: Path | None,
    tide_model_dir: Path,
    model_name: str,
    model_format: str,
    start: datetime.datetime,
    end: datetime.datetime,
    step_minutes: int,
    distance_samples: int,
) -> None:
    """Compute high-tide peak timestamps for each site and write them to JSON."""

    # -----------------------
    # 1) Load site geometries
    # -----------------------
    gdf = gpd.read_file(sites_path)

    skipped_col = "skipped"
    id_col = "Site code"

    # Filter out skipped sites if the column exists.
    if skipped_col in gdf.columns:
        # Keep rows where skipped == False
        gdf = gdf[gdf[skipped_col] == False].copy()  # noqa: E712

    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.centroid

    if id_col not in gdf.columns:
        raise click.ClickException(
            f"id-col '{id_col}' not found in input. Available columns: {list(gdf.columns)}"
        )

    # -----------------------
    # 2) Build time grid
    # -----------------------

    if end <= start:
        raise click.ClickException("--end must be after --start")

    minutes = np.arange(
        start,
        end,
        np.timedelta64(step_minutes, "m"),
    ).astype("datetime64[ns]")

    if minutes.size == 0:
        raise click.ClickException("Time grid is empty; check --start/--end/--step-minutes")

    # -----------------------
    # 3) Compute tide series
    # -----------------------
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()

    click.echo(
        f"Computing tide elevations for {len(gdf)} sites over {len(minutes)} timesteps "
        f"({step_minutes}-minute cadence)"
    )

    elev = calc_tide_elevatons(
        xs,
        ys,
        minutes,
        tide_model_dir,
        model_name,
        model_format,
    )

    # `elev` is expected to be shape (N_sites, T)
    if elev.shape[0] != len(gdf):
        raise click.ClickException(
            f"Unexpected elevation shape {elev.shape}; expected first dim == {len(gdf)}"
        )

    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            elev=elev,
            minutes=minutes,
            site_ids=gdf[id_col].astype(str).to_numpy(),
            x=xs,
            y=ys,
            model_name=model_name,
            model_format=model_format,
            step_minutes=step_minutes,
        )
        click.echo(f"Wrote full elevation array -> {out_npz}")

    # -----------------------
    # 4) Find peaks per site
    # -----------------------
    grid_peaks: dict[str, list[str]] = {}

    for i, row in gdf.reset_index(drop=True).iterrows():
        site_id = str(row[id_col])

        # Find peaks on this site's tide signal.
        # `distance` is measured in samples (not minutes/hours).
        peaks, _props = find_peaks(
            elev[i],  # type: ignore
            distance=distance_samples,
        )

        # Convert datetime64 to ISO strings for JSON portability.
        # Use numpy's string conversion (ISO-like) for each timestamp.
        ts_list = [str(t) for t in minutes[peaks]]
        grid_peaks[site_id] = ts_list

    # -----------------------
    # 5) Write output JSON
    # -----------------------
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(grid_peaks, f, indent=2)

    click.echo(f"Wrote peaks for {len(grid_peaks)} sites -> {out_json}")


if __name__ == "__main__":
    main()
