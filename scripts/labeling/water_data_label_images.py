from datetime import datetime
from pathlib import Path

import click
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from PIL import Image

from estuary.util.img import tif_to_rgb


def filter_polars_by_date(
    df: pl.DataFrame, time_col: str, start_dt: datetime, end_dt: datetime
) -> pl.DataFrame:
    """Return df filtered between start_dt and end_dt (inclusive) on time_col. If both are None,
    return df unmodified."""
    mask = (pl.col(time_col) >= pl.lit(start_dt)) & (pl.col(time_col) <= pl.lit(end_dt))
    return df.filter(mask)


def contiguous_segments(
    df: pd.DataFrame, time_col: str = "acquired", gap_hours: int = 24, days: int = 3
) -> list[pd.DataFrame]:
    """
    Find all contiguous segments in a time series (gaps > gap_hours separate segments).
    Return only the longest contiguous segment (largest time span).
    """
    out = []

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    if df.empty:
        return []

    gap = pd.Timedelta(hours=gap_hours)
    seg_id = (df[time_col].diff() > gap).cumsum()
    # Find all segments
    for sid in seg_id.unique():
        seg = df[seg_id == sid]

        # drop the LAST N calendar days
        last_ts = seg[time_col].max()
        stop = last_ts - pd.Timedelta(days=days)
        seg = seg[seg[time_col] < stop]

        if seg.empty:
            continue
        out.append(seg.copy())

    return out


def format_time_axis(ax: Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def load_raw_logger_data(pth: Path) -> pl.DataFrame:
    empa = pl.read_csv(pth)

    # Normalize unit strings (strip whitespace) so comparisons match exactly
    empa = empa.with_columns(
        [
            pl.col("raw_conductivity_unit")
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("raw_conductivity_unit"),
            pl.col("raw_depth_unit").cast(pl.Utf8).str.strip_chars().alias("raw_depth_unit"),
        ]
    )

    # Convert μS/cm → mS/cm and overwrite both columns (cast numeric first)
    empa = empa.with_columns(
        [
            pl.when(pl.col("raw_conductivity_unit") == "uS/cm")
            .then(pl.col("raw_conductivity").cast(pl.Float64, strict=False) / 1000.0)
            .when(pl.col("raw_conductivity_unit") == "mS/cm")
            .then(pl.col("raw_conductivity").cast(pl.Float64, strict=False))
            .otherwise(pl.col("raw_conductivity").cast(pl.Float64, strict=False))
            .alias("raw_conductivity"),
            # Only standardize the unit to mS/cm when the original was a known conductivity unit
            pl.when(pl.col("raw_conductivity_unit").is_in(["uS/cm", "mS/cm"]))
            .then(pl.lit("mS/cm"))
            .otherwise(pl.col("raw_conductivity_unit"))
            .alias("raw_conductivity_unit"),
        ]
    )

    # Convert cm → m and overwrite both columns (cast numeric first)
    empa = empa.with_columns(
        [
            pl.when(pl.col("raw_depth_unit") == "cm")
            .then(pl.col("raw_depth").cast(pl.Float64, strict=False) / 100.0)
            .when(pl.col("raw_depth_unit") == "m")
            .then(pl.col("raw_depth").cast(pl.Float64, strict=False))
            .otherwise(pl.col("raw_depth").cast(pl.Float64, strict=False))
            .alias("raw_depth"),
            pl.when(pl.col("raw_depth_unit").is_in(["cm", "m"]))
            .then(pl.lit("m"))
            .otherwise(pl.col("raw_depth_unit"))
            .alias("raw_depth_unit"),
        ]
    )

    # define time parsing (try multiple formats)
    parsed_dt = pl.coalesce(
        [
            pl.col("samplecollectiontimestamp").str.strptime(
                pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False
            ),
            pl.col("samplecollectiontimestamp").str.strptime(
                pl.Datetime, "%d/%m/%Y %H:%M:%S%.f", strict=False
            ),
        ]
    )

    # offsets relative to UTC (Polars doesn’t know “PST/PDT” by name)
    # PST = UTC−8, PDT = UTC−7
    empa = empa.with_columns([parsed_dt.alias("samplecollectiontimestamp_parsed")])

    # apply offset based on timezone
    empa = empa.with_columns(
        [
            pl.when(pl.col("samplecollectiontimezone") == "PST")
            .then(pl.col("samplecollectiontimestamp_parsed") + pl.duration(hours=8))  # PST -> UTC
            .when(pl.col("samplecollectiontimezone") == "PDT")
            .then(pl.col("samplecollectiontimestamp_parsed") + pl.duration(hours=7))  # PDT -> UTC
            .when(pl.col("samplecollectiontimezone") == "UTC")
            .then(pl.col("samplecollectiontimestamp_parsed"))
            .otherwise(pl.col("samplecollectiontimestamp_parsed"))
            .alias("samplecollectiontimestamp_utc2")
        ]
    )

    return empa


def filter_raw_depth(sensor_data: pl.DataFrame, other_cols: list[str]) -> pl.DataFrame | None:
    col1 = "raw_depth_unit"
    col2 = "raw_depth_qcflag"
    col3 = "raw_depth"
    if col1 in sensor_data.columns:
        if "Not Recorded" in sensor_data[col1].unique():
            return
        else:
            d = sensor_data[col3].cast(pl.Float64, strict=False)
            all_na = d.null_count() == d.len()
            all_same = d.drop_nulls().n_unique() <= 1

            if all_na or all_same:
                return

            return sensor_data[[*other_cols, col3]]
    else:
        s2 = sensor_data.filter(pl.col(col2).is_finite())  # type: ignore
        valid_rows = sensor_data[col3].cast(pl.Float64, strict=False).is_finite().sum()
        if len(sensor_data) == len(s2) and len(sensor_data) == valid_rows:
            return sensor_data[[*other_cols, col3]]

    return None


def load_corrected_water_data(pth: Path) -> pl.DataFrame:
    corr = pl.read_csv(pth)
    corr = corr.with_columns(
        [
            pl.coalesce(
                [
                    pl.col("samplecollectiontimestamp").str.strptime(
                        pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False
                    ),
                    pl.col("samplecollectiontimestamp").str.strptime(
                        pl.Datetime, "%d/%m/%Y %H:%M:%S%.f", strict=False
                    ),
                ]
            ).alias("samplecollectiontimestamp_utc2")
        ]
    )

    return corr


def remove_outliers(
    df: pd.DataFrame,
    depth_col: str,
    rolling_window: str = "1h",
    drop_threshold: float = 0.2,
    max_len_points: int = 20,
) -> pd.DataFrame:
    depth_df = df.copy()

    # Parameters you can tune
    # rolling_window: local baseline window (1 hour)
    # drop_threshold: depth drop threshold (in same units as raw_depth)
    # max_len_points: max length (in samples) of a "short" invalid drop (e.g. 20 * 6min = 2h)

    # 1. Rolling median baseline
    baseline = depth_df[depth_col].rolling(rolling_window, center=True, min_periods=1).median()

    # 2. Residual (how far below/above local baseline)
    residual = depth_df[depth_col] - baseline

    # 3. Boolean mask for drops: residual much LOWER than baseline
    drop_mask = residual < -drop_threshold

    # 4. Identify contiguous segments of drops
    # Each run of True values gets a unique group id
    group_id = (drop_mask != drop_mask.shift()).cumsum()

    # Count length of each group
    group_sizes = drop_mask.groupby(group_id).sum()  # sum works since True=1, False=0

    # Find group ids representing short drop segments
    short_drop_groups = group_sizes[(group_sizes > 0) & (group_sizes <= max_len_points)].index  # type: ignore

    # Final mask: only points that are part of short drop segments
    short_drop_mask = drop_mask & group_id.isin(short_drop_groups)

    # 5. Set these to NaN
    depth_df.loc[short_drop_mask, depth_col] = np.nan

    # 6. Interpolate over them (time-based)
    depth_df[depth_col] = depth_df[depth_col].interpolate(method="time")

    return depth_df


def plot_depth_centered_on_event(
    df,
    depth_col: str,
    event_ts,
    out_path: Path,
):
    # Resolve time series x-axis
    x = df.index

    # Resolve event timestamp
    event_ts = pd.to_datetime(event_ts)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, df[depth_col].values, linewidth=1)
    ax.axvline(event_ts, linestyle="--", linewidth=1, color="red")  # event marker

    ax.set_xlabel("Time")
    ax.set_ylabel(depth_col)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def tif_to_jpeg(
    tif_path: Path,
    dest_path: Path,
    jpeg_quality: int = 95,
) -> None:
    """Convert a GeoTIFF to RGB JPEG using estuary.util.tif_to_rgb()."""
    rgb = tif_to_rgb(tif_path)
    if np.all(rgb == 0):
        return
    img = Image.fromarray(rgb)
    img.save(dest_path, format="JPEG", quality=jpeg_quality, optimize=True)
    img.close()


@click.command()
@click.option(
    "-l",
    "--labels-path",
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to existing labels.",
)
@click.option(
    "-r",
    "--regions-path",
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to regions info.",
)
@click.option(
    "-rd",
    "--raw-water-logs-path",
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to logger-raw-publish.csv.",
)
@click.option(
    "-cd",
    "--corrected-water-logs-path",
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to logger-raw-depth-correction-publish.csv.",
)
@click.option(
    "-s",
    "--save-dir",
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Directory to save the results.",
)
@click.option(
    "--min-segment-length",
    type=int,
    required=True,
    default=60,
    help="Minimum segment length in days.",
)
def main(
    labels_path: Path,
    regions_path: Path,
    raw_water_logs_path: Path,
    corrected_water_logs_path: Path,
    save_dir: Path,
    min_segment_length: int,
):
    regions_gdf = gpd.read_file(regions_path).rename(columns={"Site code": "region"})
    regions_gdf = regions_gdf[
        (~regions_gdf.skipped) & (~regions_gdf.potential_add) & (~regions_gdf.siteid.isna())
    ].copy()
    regions_gdf = regions_gdf.set_index("region")

    labels = pd.read_csv(labels_path)
    labels["acquired"] = pd.to_datetime(labels["acquired"], errors="coerce")
    labels = labels.sort_values(by=["region", "acquired"])
    labels = labels[(~labels["acquired"].isna()) & (labels["label"] != "unsure")].copy()

    # "/Volumes/x10pro/estuary/ca_all/empa/logger-raw-publish.csv"
    raw_water_data = load_raw_logger_data(raw_water_logs_path)

    # "/Volumes/x10pro/estuary/ca_all/empa/logger-raw-depth-correction-publish.csv"
    corrected_water_data = load_corrected_water_data(corrected_water_logs_path)

    print(
        corrected_water_data.filter(pl.col("siteid") == "SC-MAL")
        .filter(pl.col("sensorid") == "X2274")
        .slice(19006, 10)
    )
    print(
        raw_water_data.filter(pl.col("siteid") == "SC-MAL")
        .filter(pl.col("sensorid") == "X2274")
        .slice(19006, 10)
    )

    rl = labels[
        (labels.region == 2145) & (labels["acquired"] > pd.Timestamp(year=2023, day=4, month=7))
    ]
    print(rl)

    valid_segments = []
    segments_meta = []

    base_columns = [
        "samplecollectiontimestamp_utc2",
        "estuaryname",
        "siteid",
        "sensortype",
        "sensorid",
    ]

    for region, row in regions_gdf.iterrows():
        siteid = row.siteid

        corr = corrected_water_data.filter(pl.col("siteid") == siteid)
        raw = raw_water_data.filter(pl.col("siteid") == siteid)
        region_labels = labels[labels.region == region]
        if region_labels.empty:
            continue
        start_date = region_labels["acquired"].min()
        end_date = region_labels["acquired"].max()

        # Convert to naive UTC datetimes for comparison
        start_dt = start_date.replace(tzinfo=None)
        end_dt = end_date.replace(tzinfo=None)

        # Filter Polars DataFrames via helper
        raw = filter_polars_by_date(raw, "samplecollectiontimestamp_utc2", start_dt, end_dt)
        corr = filter_polars_by_date(corr, "samplecollectiontimestamp_utc2", start_dt, end_dt)

        # Filter nan and null
        raw = raw.filter(pl.col("raw_depth").is_not_null() & ~pl.col("raw_depth").is_nan())
        corr = corr.filter(
            pl.col("corrected_depth").is_not_null() & ~pl.col("corrected_depth").is_nan()
        )

        raw_sensors = raw.select("sensorid").unique().to_series().to_list()
        corr_sensors = corr.select("sensorid").unique().to_series().to_list()

        for sensor in corr_sensors:
            sensor_data = corr.filter(pl.col("sensorid") == sensor)
            sensor_df = (
                sensor_data[["corrected_depth", *base_columns]]
                .to_pandas()
                .rename(columns={"samplecollectiontimestamp_utc2": "acquired"})
            )
            segments = contiguous_segments(sensor_df)

            for segment in segments:
                segment = segment.set_index("acquired")
                segment = remove_outliers(segment, "corrected_depth")
                smn = segment.index.min()
                smx = segment.index.max()
                dur = smx - smn
                if dur < pd.Timedelta(days=min_segment_length):
                    continue

                segments_meta.append(
                    {
                        "region": region,
                        "sensor": sensor,
                        "depth_col": "corrected_depth",
                        "start_dt": smn,
                        "end_dt": smx,
                        "duration": dur,
                        "valid_segment_index": len(valid_segments),
                    }
                )
                valid_segments.append(segment)

        for sensor in raw_sensors:
            sensor_data = raw.filter(pl.col("sensorid") == sensor)
            filter_raw_depth(sensor_data, base_columns)
            sensor_df = (
                sensor_data[["raw_depth", *base_columns]]
                .to_pandas()
                .rename(columns={"samplecollectiontimestamp_utc2": "acquired"})
            )
            segments = contiguous_segments(sensor_df)

            for segment in segments:
                segment = segment.set_index("acquired")
                segment = remove_outliers(segment, "raw_depth")
                smn = segment.index.min()
                smx = segment.index.max()
                dur = smx - smn
                if dur < pd.Timedelta(days=min_segment_length):
                    continue

                segments_meta.append(
                    {
                        "region": region,
                        "sensor": sensor,
                        "depth_col": "raw_depth",
                        "start_dt": smn,
                        "end_dt": smx,
                        "duration": dur,
                        "valid_segment_index": len(valid_segments),
                    }
                )
                valid_segments.append(segment)

    save_dir.mkdir(exist_ok=True, parents=True)
    segments_df = pd.DataFrame(segments_meta)
    segments_df = segments_df.sort_values(by=["region", "depth_col", "sensor", "start_dt"])
    segments_df.to_csv(save_dir / "segments.csv", index=False)

    to_label = []

    for region, _ in regions_gdf.iterrows():
        region_labels = labels[labels.region == region]

        region_segments = segments_df[segments_df.region == region]
        if region_segments.empty:
            continue

        for _, label_row in region_labels.iterrows():
            acquired: pd.Timestamp = label_row["acquired"]
            source_tif = Path(label_row["source_tif"])
            if not source_tif.exists():
                print("WARNING: Missing tif", source_tif)
                continue

            l_segments = region_segments[
                (region_segments.start_dt < acquired - pd.Timedelta(days=min_segment_length // 2))
                & (region_segments.end_dt > acquired + pd.Timedelta(days=min_segment_length // 2))
            ]
            if not len(l_segments):
                continue

            best_sensor_segment = l_segments.iloc[0]
            best_idx = best_sensor_segment.valid_segment_index
            segment = valid_segments[best_idx]  # type: ignore
            start_dt = acquired - pd.Timedelta(days=min_segment_length // 2)
            end_dt = acquired + pd.Timedelta(days=min_segment_length // 2)
            disp_segment = segment[(segment.index > start_dt) & (segment.index < end_dt)]

            depth_col = best_sensor_segment.depth_col

            fig_path = save_dir / "plots" / str(region) / f"{source_tif.stem}.jpeg"
            fig_path.parent.mkdir(exist_ok=True, parents=True)
            plot_depth_centered_on_event(disp_segment, depth_col, acquired, fig_path)

            img_path = save_dir / "images" / str(region) / f"{source_tif.stem}.jpeg"
            img_path.parent.mkdir(exist_ok=True, parents=True)
            tif_to_jpeg(source_tif, img_path)

            to_label.append(
                {
                    "region": region,
                    "orig_label": label_row["label"],
                    "source_tif": source_tif,
                    "img_path": img_path,
                    "fig_path": fig_path,
                }
            )

    pd.DataFrame(to_label).to_csv(save_dir / "to_label.csv", index=False)

    print("Done!")


if __name__ == "__main__":
    main()
