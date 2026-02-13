from datetime import datetime
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import tqdm


def remap_df(df: pd.DataFrame, height_col: str, region: int, source: str) -> pd.DataFrame:
    # {
    #     "timestamp_utc": ts_dt,
    #     "height": height_m,
    #     "region": region,
    #     "source": "flot",
    #     "sensor_id": device_id,
    # }

    df = (
        df.reset_index()
        .rename(
            columns={
                height_col: "height",
                "sensorid": "sensor_id",
                "": "timestamp_utc",
                "acquired": "timestamp_utc",
            }
        )
        .drop(columns=["estuaryname", "sensortype", "siteid"])
    )
    df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")  # type: ignore
    df["region"] = region
    df["source"] = source
    df["sensor_id"] = df.sensor_id.astype("string")

    return df


def filter_by_date(
    df: pl.DataFrame, time_col: str, start_dt: datetime, end_dt: datetime
) -> pl.DataFrame:
    """Return df filtered between start_dt and end_dt (inclusive) on time_col. If both are None,
    return df unmodified."""
    mask = (pl.col(time_col) >= pl.lit(start_dt)) & (pl.col(time_col) <= pl.lit(end_dt))
    return df.filter(mask)


def contiguous_segments(
    df: pd.DataFrame, time_col: str = "acquired", gap_hours: int = 48
) -> list[pd.DataFrame]:
    """
    Find all contiguous segments in a time series (gaps > gap_hours separate segments).
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

        if seg.empty:
            continue
        out.append(seg.copy())

    return out


def load_emailed_logger_data(pth: Path) -> pl.DataFrame:
    empa = pl.read_csv(pth)
    if "sensorid" in empa.columns:
        empa = empa.with_columns(pl.col("sensorid").cast(pl.Utf8))

    # Clean the string first (handles "7/22/15 0:00 " etc.)
    ts_str = pl.col("samplecollectiontimestamp").cast(pl.Utf8).str.strip_chars()

    parsed_dt = pl.coalesce(
        [
            # 2-digit year 24-hour  ✅ this is the one that *should* catch 7/22/15 0:00
            ts_str.str.strptime(pl.Datetime, "%m/%d/%y %H:%M", strict=False),
            # ts_str.str.strptime(pl.Datetime, "%m/%d/%y %H:%M:%S", strict=False),
            # 2-digit year 12-hour with AM/PM
            # ts_str.str.strptime(pl.Datetime, "%m/%d/%y %I:%M %p", strict=False),
            # ts_str.str.strptime(pl.Datetime, "%m/%d/%y %I:%M:%S %p", strict=False),
            # final fallback: let Polars infer common formats (handles non-zero-padded cases)
            # ts_str.str.to_datetime(strict=False),
            # 4-digit year 24-hour
            ts_str.str.strptime(pl.Datetime, "%m/%d/%Y %H:%M", strict=False),
            # ts_str.str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False),
            # 4-digit year 12-hour with AM/PM
            ts_str.str.strptime(pl.Datetime, "%m/%d/%Y %I:%M %p", strict=False),
            ts_str.str.strptime(pl.Datetime, "%m/%d/%Y %I:%M:%S %p", strict=False),
        ]
    )

    empa = empa.with_columns([parsed_dt.alias("samplecollectiontimestamp_parsed")])

    # apply offset based on timezone
    empa = empa.with_columns(
        [
            pl.when(pl.col("time_units") == "PST")
            .then(pl.col("samplecollectiontimestamp_parsed") + pl.duration(hours=8))  # PST -> UTC
            .when(pl.col("time_units") == "PDT")
            .then(pl.col("samplecollectiontimestamp_parsed") + pl.duration(hours=7))  # PDT -> UTC
            .when(pl.col("time_units") == "UTC")
            .then(pl.col("samplecollectiontimestamp_parsed"))
            .otherwise(pl.col("samplecollectiontimestamp_parsed"))
            .alias("samplecollectiontimestamp_utc2")
        ]
    )

    return empa


def load_raw_logger_data(pth: Path) -> pl.DataFrame:
    empa = pl.read_csv(pth)
    if "sensorid" in empa.columns:
        empa = empa.with_columns(pl.col("sensorid").cast(pl.Utf8))

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
    if "sensorid" in corr.columns:
        corr = corr.with_columns(pl.col("sensorid").cast(pl.Utf8))
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


@click.command()
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
    "-ed",
    "--emailed-water-logs-path",
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to additional logger data csv.",
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
    regions_path: Path,
    raw_water_logs_path: Path,
    emailed_water_logs_path: Path,
    corrected_water_logs_path: Path,
    save_dir: Path,
    min_segment_length: int,
):
    regions_gdf = gpd.read_file(regions_path).rename(columns={"Site code": "region"})
    regions_gdf = regions_gdf[(~regions_gdf.skipped) & (~regions_gdf.siteid.isna())].copy()
    regions_gdf = regions_gdf.set_index("region")

    # "/Volumes/x10pro/estuary/water_data/raw/empa/logger-raw-publish.csv"
    raw_water_data = load_raw_logger_data(raw_water_logs_path)

    # "/Volumes/x10pro/estuary/water_data/raw/empa/emailed_water_data.csv"
    emailed_water_data = load_emailed_logger_data(emailed_water_logs_path)

    # "/Volumes/x10pro/estuary/water_data/raw/empa/logger-raw-depth-correction-publish.csv"
    corrected_water_data = load_corrected_water_data(corrected_water_logs_path)

    base_columns = [
        "samplecollectiontimestamp_utc2",
        "estuaryname",
        "siteid",
        "sensortype",
        "sensorid",
    ]

    for region, row in tqdm.tqdm(regions_gdf.iterrows(), total=len(regions_gdf)):
        region_valid_segments: list[pd.DataFrame] = []
        siteid = row.siteid

        corr = corrected_water_data.filter(pl.col("siteid") == siteid)
        raw = raw_water_data.filter(pl.col("siteid") == siteid)
        emailed = emailed_water_data.filter(pl.col("siteid") == siteid)

        # Convert to naive UTC datetimes for comparison
        start_dt = datetime(year=2017, month=1, day=1).replace(tzinfo=None)
        end_dt = datetime(year=2026, month=1, day=1).replace(tzinfo=None)

        # Filter Polars DataFrames via helper
        raw = filter_by_date(raw, "samplecollectiontimestamp_utc2", start_dt, end_dt)
        emailed = filter_by_date(emailed, "samplecollectiontimestamp_utc2", start_dt, end_dt)
        corr = filter_by_date(corr, "samplecollectiontimestamp_utc2", start_dt, end_dt)

        # Filter nan and null
        raw = raw.filter(pl.col("raw_depth").is_not_null() & ~pl.col("raw_depth").is_nan())
        emailed = emailed.filter(pl.col("depth_m").is_not_null() & ~pl.col("depth_m").is_nan())
        corr = corr.filter(
            pl.col("corrected_depth").is_not_null() & ~pl.col("corrected_depth").is_nan()
        )

        raw_sensors = raw.select("sensorid").unique().to_series().to_list()
        emailed_sensors = emailed.select("sensorid").unique().to_series().to_list()
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

                region_valid_segments.append(
                    remap_df(segment, "corrected_depth", int(region), "corrected_depth_empa")  # type: ignore
                )

        for sensor in raw_sensors:
            sensor_data = raw.filter(pl.col("sensorid") == sensor)
            sensor_data = filter_raw_depth(sensor_data, base_columns)
            if sensor_data is None:
                continue

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

                region_valid_segments.append(
                    remap_df(segment, "raw_depth", int(region), "raw_depth_empa")  # type: ignore
                )

        for sensor in emailed_sensors:
            sensor_data = emailed.filter(pl.col("sensorid") == sensor)

            sensor_df = (
                sensor_data[["depth_m", *base_columns]]
                .to_pandas()
                .rename(columns={"samplecollectiontimestamp_utc2": "acquired"})
            )
            segments = contiguous_segments(sensor_df)

            for segment in segments:
                segment = segment.set_index("acquired")
                segment = remove_outliers(segment, "depth_m")
                smn = segment.index.min()
                smx = segment.index.max()
                dur = smx - smn
                if dur < pd.Timedelta(days=min_segment_length):
                    continue

                region_valid_segments.append(
                    remap_df(segment, "depth_m", int(region), "emailed_depth_empa")  # type: ignore
                )

        if len(region_valid_segments):
            df = pd.concat(region_valid_segments, ignore_index=True)

            save_path = save_dir / str(region) / "empa.csv"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(save_path, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
