#!/usr/bin/env python3

from pathlib import Path

import click
import numpy as np
import pandas as pd

from estuary.drivers.tides import calc_tide_elevatons


@click.command()
@click.option(
    "--points-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to CSV or Parquet file containing nearest tide points.",
)
@click.option(
    "--tide-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the base tide model directory.",
)
@click.option(
    "--definition-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a pyTMD definition file.",
)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output parquet file.",
)
@click.option(
    "--start",
    default="2017-01-01",
    show_default=True,
    help="Start timestamp for tide calculation.",
)
@click.option(
    "--end",
    default="2025-01-01",
    show_default=True,
    help="End timestamp for tide calculation.",
)
@click.option(
    "--freq",
    default="30min",
    show_default=True,
    help="Pandas frequency string for the tide time series.",
)
@click.option(
    "--crop-buffer-deg",
    default=0.05,
    show_default=True,
    type=float,
    help="Crop buffer passed to calc_tide_elevatons.",
)
def main(
    points_path: Path,
    tide_dir: Path,
    definition_file: Path,
    out_path: Path,
    start: str,
    end: str,
    freq: str,
    crop_buffer_deg: float,
) -> None:
    """Calculate tidal elevations for a set of preselected tide points."""
    suffix = points_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(points_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(points_path)
    else:
        raise click.ClickException("points-path must be a .csv or .parquet file")

    required_columns = {"site_id", "pt_lon_360", "pt_lat"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise click.ClickException(f"Points file is missing required columns: {sorted(missing)}")

    ts = pd.date_range(start=start, end=end, freq=freq).to_numpy().astype("datetime64[ns]")
    if len(ts) == 0:
        raise click.ClickException("Time range produced zero timestamps")

    xs = df["pt_lon_360"].to_numpy()
    ys = df["pt_lat"].to_numpy()

    tides = calc_tide_elevatons(
        xs=xs,
        ys=ys,
        ts=ts,
        model_directory=tide_dir,
        definition_file=definition_file,
        crop_buffer_deg=crop_buffer_deg,
    )

    # convert to long format
    site_ids = df["site_id"].to_numpy()

    long_df = pd.DataFrame(
        {
            "site_id": np.repeat(site_ids, len(ts)),
            "time": np.tile(ts, len(site_ids)),
            "tide_elevation": tides.reshape(-1),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_path)

    click.echo(
        f"Wrote tides for {len(site_ids)} sites and {len(ts)} timestamps "
        f"({len(long_df)} rows) to {out_path}"
    )


if __name__ == "__main__":
    main()
