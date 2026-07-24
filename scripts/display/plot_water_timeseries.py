#!/usr/bin/env python3
"""Save one water-level time-series plot centered on a satellite image.

Example:
    uv run python scripts/display/plot_water_timeseries.py \
        --tif /path/to/20170801_174611_1010_3B_AnalyticMS_SR_clip.tif \
        --output-dir /path/to/output_dir \
        --length-days 7

The image timestamp is read from the filename and the region is read from the
image's parent directory structure (``.../<region>/files/<image>.tif``).
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

from estuary.util.data import parse_dt_from_pth

DEFAULT_WATER_DATA_PATH = Path("/Volumes/x10pro/estuary/water_data/processed")


def filter_outliers(df: pd.DataFrame, depth_col: str = "height") -> pd.DataFrame:
    """Remove depth values outside a lenient three-IQR envelope."""
    values = df[depth_col].astype(float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    return df.loc[values.between(q1 - 3 * iqr, q3 + 3 * iqr)].copy()


def contiguous_segments(
    df: pd.DataFrame,
    time_col: str = "timestamp_utc",
    gap_hours: int = 48,
) -> list[pd.DataFrame]:
    """Split a sensor series wherever consecutive observations are too far apart."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    if df.empty:
        return []

    segment_ids = (df[time_col].diff() > pd.Timedelta(hours=gap_hours)).cumsum()
    return [filter_outliers(segment) for _, segment in df.groupby(segment_ids)]


def infer_region(tif_path: Path) -> int:
    """Read the region from ``.../<region>/files/<image>.tif``."""
    try:
        return int(tif_path.parent.parent.name)
    except ValueError as exc:
        raise ValueError(
            "Could not infer the region from the TIF path; expected "
            "'.../<region>/files/<image>.tif'."
        ) from exc


def load_plot_segment(
    region_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load the longest sensor segment that spans the requested plot window."""
    segments: list[pd.DataFrame] = []
    for csv_path in sorted(region_dir.glob("*.csv")):
        water = pd.read_csv(csv_path, dtype={"sensor_id": "string", "source": "string"})
        required = {"timestamp_utc", "height", "sensor_id", "source"}
        missing = required - set(water.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        for _, sensor in water.groupby(["sensor_id", "source"], dropna=False):
            segments.extend(contiguous_segments(sensor))

    covering = [
        segment
        for segment in segments
        if segment["timestamp_utc"].min() <= start and segment["timestamp_utc"].max() >= end
    ]
    if not covering:
        raise ValueError(
            f"No continuous water-level series for region {region_dir.name} spans "
            f"{start.isoformat()} through {end.isoformat()}."
        )

    segment = max(
        covering,
        key=lambda frame: frame["timestamp_utc"].max() - frame["timestamp_utc"].min(),
    )
    return (
        segment.loc[segment["timestamp_utc"].between(start, end)]
        .sort_values("timestamp_utc")
        .copy()
    )


def save_plot(
    water: pd.DataFrame,
    acquired: pd.Timestamp,
    output_path: Path,
    dpi: int,
) -> None:
    """Save a single water-level plot with the image time marked in red."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(water["timestamp_utc"], water["height"], linewidth=1)
    ax.axvline(acquired, linestyle="--", linewidth=1, color="red")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Water level (m)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option(
    "--tif",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="TIF at .../<region>/files/<YYYYMMDD_HHMMSS...>.tif.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Directory in which to save <TIF stem>.jpeg.",
)
@click.option(
    "--length-days",
    type=click.FloatRange(min=0, min_open=True),
    default=7.0,
    show_default=True,
    help="Total time-series length in days, centered on the image.",
)
@click.option(
    "--water-data-path",
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
    default=DEFAULT_WATER_DATA_PATH,
    show_default=True,
    help="Processed water-data directory.",
)
@click.option(
    "--dpi",
    type=click.IntRange(min=1),
    default=300,
    show_default=True,
    help="Output resolution.",
)
def main(
    tif: Path,
    output_dir: Path,
    length_days: float,
    water_data_path: Path,
    dpi: int,
) -> None:
    """Save one water-level plot centered on a satellite TIF acquisition."""
    tif_path = tif
    half_window = pd.Timedelta(days=length_days / 2)

    region = infer_region(tif_path)
    acquired = pd.Timestamp(parse_dt_from_pth(tif_path), tz="UTC")
    start = acquired - half_window
    end = acquired + half_window

    region_dir = water_data_path / str(region)
    if not region_dir.is_dir():
        raise click.ClickException(f"Water-data directory does not exist: {region_dir}")

    water = load_plot_segment(region_dir, start, end)
    output_path = output_dir / f"{tif_path.stem}.jpeg"
    save_plot(water, acquired, output_path, dpi)
    click.echo(f"Saved {output_path}")


if __name__ == "__main__":
    main()
