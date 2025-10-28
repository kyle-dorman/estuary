import json
from datetime import datetime
from pathlib import Path

import click
import geopandas as gpd
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import polars as pl
import rasterio
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

from estuary.model.data import parse_dt_from_pth
from estuary.util import broad_band, draw_border, false_color


def filter_polars_by_date(
    df: pl.DataFrame, time_col: str, start_dt: datetime | None, end_dt: datetime | None
) -> pl.DataFrame:
    """Return df filtered between start_dt and end_dt (inclusive) on time_col. If both are None,
    return df unmodified."""
    if not (start_dt or end_dt):
        return df
    if start_dt and end_dt:
        mask = (pl.col(time_col) >= pl.lit(start_dt)) & (pl.col(time_col) <= pl.lit(end_dt))
    elif start_dt:
        mask = pl.col(time_col) >= pl.lit(start_dt)
    else:
        mask = pl.col(time_col) <= pl.lit(end_dt)
    return df.filter(mask)


def keep_longest_contiguous(
    df: pd.DataFrame, time_col: str = "acquired", gap_hours: int = 4, days: int = 7
) -> pd.DataFrame:
    """
    Find all contiguous segments in a time series (gaps > gap_hours separate segments).
    Return only the longest contiguous segment (largest time span).
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col)
    if out.empty:
        return out.iloc[0:0]
    gap = pd.Timedelta(hours=gap_hours)
    seg_id = (out[time_col].diff() > gap).cumsum()
    # Find all segments, keep the one with the largest span
    best_idx = None
    best_span = pd.Timedelta(0)
    for sid in seg_id.unique():
        seg = out[seg_id == sid]
        if seg.empty:
            continue
        span = seg[time_col].max() - seg[time_col].min()
        if span > best_span:
            best_span = span
            best_idx = sid
    if best_idx is None:
        return out.iloc[0:0]
    out = out[seg_id == best_idx]

    # 3) drop the LAST N calendar days (data “gets wonky at the end”)
    last_ts = out[time_col].max()
    stop = last_ts - pd.Timedelta(days=days)
    out = out[out[time_col] < stop]

    return out


def _find_gaps(
    df_time_sorted: pd.DataFrame,
    time_col: str,
    min_gap: pd.Timedelta,
    edge_buffer: pd.Timedelta,
    series_start: pd.Timestamp | None = None,
    series_end: pd.Timestamp | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return a list of (gap_start, gap_end) where time delta between consecutive
    samples exceeds `min_gap`. Each interval is trimmed by `edge_buffer` on both ends.
    If `series_start`/`series_end` are provided, treat the time before the first
    sample and after the last sample as potential gaps as well.
    """
    if df_time_sorted.empty:
        return []

    ts = df_time_sorted[time_col].sort_values().reset_index(drop=True)
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    # Leading gap: from series_start to first timestamp
    if series_start is not None:
        lead = ts.iloc[0] - series_start
        if lead > min_gap:
            start = series_start
            end = ts.iloc[0] - edge_buffer
            assert start < end
            gaps.append((start, end))

    # Internal gaps
    diffs = ts.diff()
    for i in range(1, len(ts)):
        if diffs.iloc[i] is not pd.NaT and diffs.iloc[i] > min_gap:  # type: ignore[operator]
            start_raw = ts.iloc[i - 1]
            end_raw = ts.iloc[i]
            start = start_raw + edge_buffer
            end = end_raw - edge_buffer
            assert start < end
            gaps.append((start, end))

    # Trailing gap: from last timestamp to series_end
    if series_end is not None:
        trail = series_end - ts.iloc[-1]
        if trail > min_gap:
            start = ts.iloc[-1] + edge_buffer
            end = series_end
            assert start < end
            gaps.append((start, end))

    return gaps


def format_time_axis(ax: Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def find_state_changes(
    df: pd.DataFrame, state_col: str, include_first: bool = True
) -> pd.DataFrame:
    s = df[state_col].astype("int64")
    changed = s.ne(s.shift(1))
    if not include_first and len(changed):
        changed.iloc[0] = False
    return df.loc[changed, ["acquired", state_col]].rename(columns={state_col: "new_state"})


def plot_metric(
    sensor_df: pd.DataFrame,
    predictions: pd.DataFrame,
    col: str,
    pred_col: str,
    ax: Axes,
) -> None:
    if sensor_df.empty:
        return
    # predictions: ['acquired','y_true','y_prob','y_pred','region']
    # sensor_df:   ['acquired', col]

    # change points
    changes = find_state_changes(predictions, pred_col, include_first=True)

    # nearest join to grab metric value at change times
    cp = pd.merge_asof(
        changes.sort_values("acquired"),
        sensor_df.sort_values("acquired"),
        on="acquired",
        direction="nearest",
    )

    # --- compute data gaps from the predictions series ---
    gaps = _find_gaps(
        predictions[["acquired"]].sort_values("acquired"),
        time_col="acquired",
        min_gap=pd.Timedelta(days=4.5),
        edge_buffer=pd.Timedelta(hours=12),
        series_start=sensor_df["acquired"].min(),
        series_end=sensor_df["acquired"].max(),
    )

    # -------- Plot --------
    ax.plot(sensor_df["acquired"], sensor_df[col], lw=1.5, color="k")
    ax.set_title(col)

    ax.set_ylabel(col)

    # blue filled bands for gaps (solid, not dashed)
    for start, end in gaps:
        ax.axvspan(start, end, color="blue", alpha=0.15, linewidth=0)  # type: ignore

    for _, row in cp.iterrows():
        color = "green" if int(row["new_state"]) == 1 else "red"
        ax.axvline(row["acquired"], linestyle="--", color=color, alpha=0.8, linewidth=1.25)
        ax.scatter(row["acquired"], row[col], color=color, s=35, zorder=3)

    format_time_axis(ax)


def rolling_smooth(df: pd.DataFrame, time_col: str, prob_col: str, days: int = 2):
    hours = days * 24 + 4
    delta = pd.Timedelta(hours=hours)
    g = df[[time_col, prob_col]].copy()
    g[time_col] = pd.to_datetime(g[time_col], errors="coerce")
    g = g.sort_values(time_col)
    new_col = f"rolling_{prob_col}"
    g[new_col] = g[prob_col].copy()

    for idx, row in g.iterrows():
        sd = row.acquired - delta
        ed = row.acquired + delta
        num_before = ((g.acquired > sd) & (g.acquired < row.acquired)).sum()
        if not num_before:
            continue
        cols = g[(g.acquired > sd) & (g.acquired < ed)]
        g.loc[idx, new_col] = np.mean(cols[prob_col].to_numpy()).item()  # type: ignore

    return g


@click.command()
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Filter rows from this date (inclusive)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Filter rows up to this date (inclusive)",
)
def main(start_date=None, end_date=None):
    SAVE_PATH = Path("/Volumes/x10pro/estuary/display_empa")

    skipped_regions = pd.read_csv("/Volumes/x10pro/estuary/geos/skipped_regions.csv")[
        "Site code"
    ].to_list()

    gdf = gpd.read_file("/Users/kyledorman/data/estuary/geos/ca_data_w_usgs.geojson")
    gdf = gdf[~gdf["Site code"].isin(skipped_regions)].copy()
    gdf = gdf.set_index("Site code")

    with open("/Users/kyledorman/data/estuary/geos/ca_empa_matching_sites.json") as f:
        matching_sites = json.load(f)

    empa = pl.read_csv("/Volumes/x10pro/estuary/ca_all/empa/logger-raw-publish.csv")

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
    empa = empa.filter(pl.col("siteid").is_in(list(matching_sites.values())))

    empa_ranges = (
        empa.group_by(["siteid", "sensorid"])
        .agg(
            [
                pl.col("samplecollectiontimestamp_utc2").min().alias("start"),
                pl.col("samplecollectiontimestamp_utc2").max().alias("end"),
            ]
        )
        # Filter groups where (end - start) > 90 days
        .filter((pl.col("end") - pl.col("start")) > pl.duration(days=90))
        .sort(["siteid", "sensorid"])
    )

    corr = pl.read_csv(
        "/Volumes/x10pro/estuary/ca_all/empa/logger-raw-depth-correction-publish.csv"
    )
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
    corr = corr.filter(pl.col("siteid").is_in(list(matching_sites.values())))

    corr_ranges = (
        corr.group_by(["siteid", "sensorid"])
        .agg(
            [
                pl.col("samplecollectiontimestamp_utc2").min().alias("start"),
                pl.col("samplecollectiontimestamp_utc2").max().alias("end"),
            ]
        )
        # Filter groups where (end - start) > 90 days
        .filter((pl.col("end") - pl.col("start")) > pl.duration(days=90))
        .sort(["siteid", "sensorid"])
    )

    preds_all = pd.read_csv(
        Path("/Users/kyledorman/data/results/estuary/train/20251021-151419/timeseries_preds.csv")
    )
    preds_all["acquired"] = preds_all["source_tif"].apply(lambda p: parse_dt_from_pth(Path(p)))
    # normalize times
    preds_all["acquired"] = (
        pd.to_datetime(preds_all["acquired"], errors="coerce", utc=True)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    preds_all = preds_all.sort_values("acquired").dropna(subset=["acquired"])
    preds_all["y_true_prob"] = 0.05
    preds_all.loc[preds_all.orig_label == "perched open", "y_true_prob"] = 0.6
    preds_all.loc[preds_all.orig_label == "open", "y_true_prob"] = 0.95

    # --------- Filter by start/end date if provided ---------
    if start_date or end_date:
        # Convert to naive UTC datetimes for comparison
        start_dt = start_date.replace(tzinfo=None) if start_date else None
        end_dt = end_date.replace(tzinfo=None) if end_date else None
        # Filter Polars DataFrames via helper
        empa = filter_polars_by_date(empa, "samplecollectiontimestamp_utc2", start_dt, end_dt)
        corr = filter_polars_by_date(corr, "samplecollectiontimestamp_utc2", start_dt, end_dt)
        # Filter pandas DataFrame
        if start_dt and end_dt:
            mask = (preds_all["acquired"] >= start_dt) & (preds_all["acquired"] <= end_dt)
        elif start_dt:
            mask = preds_all["acquired"] >= start_dt
        elif end_dt:
            mask = preds_all["acquired"] <= end_dt
        else:
            mask = slice(None)
        preds_all = preds_all.loc[mask].copy()

    empa_cols = [
        "depth",
        "h2otemp",
        "ph",
        "conductivity",
        "turbidity",
        "do",
        "chlorophyll",
        "orp",
    ]

    corr_skip = []

    for siteid, sensorid, _, _ in empa_ranges.iter_rows():
        region = next(k for k, v in matching_sites.items() if v == siteid)
        region = int(region)
        region_name = gdf.loc[region]["Site name"]
        sensor_data = empa.filter((pl.col("siteid") == siteid) & (pl.col("sensorid") == sensorid))

        keep_cols = []
        for col in empa_cols:
            col1 = f"raw_{col}_unit"
            col2 = f"raw_{col}_qcflag"
            col3 = f"raw_{col}"
            if col1 in sensor_data.columns:
                if "Not Recorded" in sensor_data[col1].unique():
                    continue
                else:
                    d = sensor_data[col3].cast(pl.Float64, strict=False)
                    # convert to Series
                    series = d.to_pandas()

                    all_na = series.isna().all()
                    all_same = series.nunique(dropna=True) <= 1

                    if all_na or all_same:
                        continue
                    keep_cols.append(col3)
            else:
                s2 = sensor_data.filter(pl.col(col2).is_finite())  # type: ignore
                valid_rows = sensor_data[col3].cast(pl.Float64, strict=False).is_finite().sum()
                if len(sensor_data) == len(s2) and len(sensor_data) == valid_rows:
                    keep_cols.append(col3)

        if not len(keep_cols):
            continue

        empa_sensor_data = (
            sensor_data[["samplecollectiontimestamp_utc2", *keep_cols]]
            .to_pandas()
            .rename(columns={"samplecollectiontimestamp_utc2": "acquired"})
            .sort_values(["acquired"])
        )
        for col in keep_cols:
            empa_sensor_data[col] = empa_sensor_data[col].apply(pd.to_numeric)

        # Merge corrected data if possible
        corr_sensor_data = (
            corr.filter((pl.col("siteid") == siteid) & (pl.col("sensorid") == sensorid))[
                ["samplecollectiontimestamp_utc2", "corrected_depth"]
            ]
            .to_pandas()
            .rename(columns={"samplecollectiontimestamp_utc2": "acquired"})
            .sort_values(["acquired"])
        )
        corr_sensor_data["corrected_depth"] = corr_sensor_data["corrected_depth"].apply(
            pd.to_numeric
        )
        corr_sensor_data = corr_sensor_data[~corr_sensor_data.isna()].copy()

        if len(corr_sensor_data):
            corr_skip.append((siteid, sensorid))
            empa_sensor_data = pd.merge(
                empa_sensor_data, corr_sensor_data, on="acquired", how="outer"
            )
            empa_sensor_data = empa_sensor_data[~empa_sensor_data.isna().any(axis=1)].copy()
            keep_cols.append("corrected_depth")

        empa_sensor_data["acquired"] = (
            pd.to_datetime(empa_sensor_data["acquired"], errors="coerce", utc=True)
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        empa_sensor_data = (
            empa_sensor_data.sort_values("acquired")
            .dropna(subset=["acquired"])
            .drop_duplicates(subset="acquired", keep="first")
        )

        # Filter gaps
        empa_sensor_data = keep_longest_contiguous(empa_sensor_data)

        date_range = empa_sensor_data["acquired"].max() - empa_sensor_data["acquired"].min()
        if date_range < pd.Timedelta(days=90):
            continue

        predictions = preds_all[preds_all.region == region]
        # keep preds within depth time span
        predictions = predictions[
            (predictions["acquired"] >= empa_sensor_data["acquired"].min())
            & (predictions["acquired"] <= empa_sensor_data["acquired"].max())
        ].copy()
        changes = find_state_changes(predictions, "y_true", include_first=False)
        if not len(changes):
            continue

        keep_cols = sorted(keep_cols)

        if not any("depth" in c for c in keep_cols):
            continue

        print(siteid, sensorid, keep_cols)

        for label_name, pred_col, prob_col in [
            ("label", "y_true", "y_true_prob"),
            ("prediction", "y_pred", "y_prob"),
        ]:
            # Plot images of changes
            changes = find_state_changes(predictions, pred_col, include_first=True)
            change_rows = predictions.loc[changes.index][["source_tif", "acquired", pred_col]]

            nrows = len(change_rows)
            fig, axes = plt.subplots(ncols=1, nrows=nrows, figsize=(7, 7 * nrows))
            if nrows == 1:
                axes = [axes]
            for ax, (_, row) in zip(axes, change_rows.iterrows(), strict=False):
                pred_color = (44, 160, 44) if row[pred_col] == 1 else (214, 39, 40)  # green/red
                pth = row.source_tif
                with rasterio.open(pth) as src:
                    data = src.read(out_dtype=np.float32)
                    nodata = src.read_masks(1) == 0
                    if len(data) == 4:
                        img = false_color(data, nodata)
                    else:
                        img = broad_band(data, nodata)
                    img = Image.fromarray(img)

                img = draw_border(img, color=pred_color)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(str(row.acquired.date()))

            fig.tight_layout()
            axes[0].set_title(
                f"{region_name} - {sensorid} - {label_name.capitalize()}", fontsize=14
            )
            file_region_name = "_".join(region_name.lower().split(" "))
            save_dir_name = label_name + "_images"
            save_path = SAVE_PATH / save_dir_name / f"{file_region_name}_{sensorid}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=200)
            plt.close(fig=fig)

            # Plot time series data
            nrows = len(keep_cols) + 1
            fig, axes = plt.subplots(ncols=1, nrows=nrows, figsize=(11, 4 * nrows), sharex=True)
            if nrows == 1:
                axes = [axes]

            for ax, col in zip(axes[: len(keep_cols)], keep_cols, strict=False):
                plot_metric(
                    sensor_df=empa_sensor_data,
                    predictions=predictions,
                    col=col,
                    pred_col=pred_col,
                    ax=ax,
                )

            # Add predicted probability plot in the last axis
            ax_prob: Axes = axes[-1]
            y_prob = rolling_smooth(predictions, "acquired", prob_col)[f"rolling_{prob_col}"]
            ax_prob.scatter(x=predictions["acquired"], y=y_prob, lw=1.2, color="gray")
            ax_prob.set_ylabel(prob_col)
            ax_prob.set_ylim(-0.05, 1.05)
            ax_prob.set_title(f"{label_name.capitalize()} probability")
            format_time_axis(ax_prob)

            axes[-1].set_xlabel("Time")
            fig.suptitle(f"{region_name} - {sensorid} - {label_name.capitalize()}", fontsize=14)
            fig.tight_layout()
            file_region_name = "_".join(region_name.lower().split(" "))
            save_path = SAVE_PATH / label_name / f"{file_region_name}_{sensorid}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=200)
            plt.close(fig=fig)

    for siteid, sensorid, _, _ in corr_ranges.iter_rows():
        key = (siteid, sensorid)
        if key in corr_skip:
            continue

        region = next(k for k, v in matching_sites.items() if v == siteid)
        region = int(region)
        region_name = gdf.loc[region]["Site name"]

        corr_sensor_data = (
            corr.filter((pl.col("siteid") == siteid) & (pl.col("sensorid") == sensorid))[
                ["samplecollectiontimestamp_utc2", "corrected_depth"]
            ]
            .to_pandas()
            .rename(columns={"samplecollectiontimestamp_utc2": "acquired"})
            .sort_values(["acquired"])
        )
        corr_sensor_data["corrected_depth"] = corr_sensor_data["corrected_depth"].apply(
            pd.to_numeric
        )
        corr_sensor_data = corr_sensor_data[~corr_sensor_data.isna()].copy()

        corr_sensor_data["acquired"] = (
            pd.to_datetime(corr_sensor_data["acquired"], errors="coerce", utc=True)
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        corr_sensor_data = (
            corr_sensor_data.sort_values("acquired")
            .dropna(subset=["acquired"])
            .drop_duplicates(subset="acquired", keep="first")
        )

        # Filter gaps
        corr_sensor_data = keep_longest_contiguous(corr_sensor_data)

        date_range = corr_sensor_data["acquired"].max() - corr_sensor_data["acquired"].min()
        if date_range < pd.Timedelta(days=90):
            continue

        predictions = preds_all[preds_all.region == region]
        # keep preds within depth time span
        predictions = predictions[
            (predictions["acquired"] >= corr_sensor_data["acquired"].min())
            & (predictions["acquired"] <= corr_sensor_data["acquired"].max())
        ].copy()
        changes = find_state_changes(predictions, "y_true", include_first=False)
        if not len(changes):
            continue

        print(siteid, sensorid, "corrected_depth")

        for label_name, pred_col, prob_col in [
            ("label", "y_true", "y_true_prob"),
            ("prediction", "y_pred", "y_prob"),
        ]:
            # Plot images of changes
            changes = find_state_changes(predictions, pred_col, include_first=True)
            change_rows = predictions.loc[changes.index][["source_tif", "acquired", pred_col]]

            nrows = len(change_rows)
            fig, axes = plt.subplots(ncols=1, nrows=nrows, figsize=(7, 7 * nrows))
            if nrows == 1:
                axes = [axes]
            for ax, (_, row) in zip(axes, change_rows.iterrows(), strict=False):
                pred_color = (44, 160, 44) if row[pred_col] == 1 else (214, 39, 40)  # green/red
                pth = row.source_tif
                with rasterio.open(pth) as src:
                    data = src.read(out_dtype=np.float32)
                    nodata = src.read_masks(1) == 0
                    if len(data) == 4:
                        img = false_color(data, nodata)
                    else:
                        img = broad_band(data, nodata)
                    img = Image.fromarray(img)

                img = draw_border(img, color=pred_color)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(str(row.acquired.date()))

            fig.tight_layout()
            axes[0].set_title(
                f"{region_name} - {sensorid} - {label_name.capitalize()}", fontsize=14
            )
            file_region_name = "_".join(region_name.lower().split(" "))
            save_dir_name = label_name + "_images"
            save_path = SAVE_PATH / save_dir_name / f"{file_region_name}_{sensorid}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=200)
            plt.close(fig=fig)

            # Plot time series data
            fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(11, 8), sharex=True)

            plot_metric(
                sensor_df=corr_sensor_data,
                predictions=predictions,
                col="corrected_depth",
                pred_col=pred_col,
                ax=axes[0],
            )

            # Add predicted probability plot in the last axis
            ax_prob: Axes = axes[-1]
            y_prob = rolling_smooth(predictions, "acquired", prob_col)[f"rolling_{prob_col}"]
            ax_prob.scatter(x=predictions["acquired"], y=y_prob, lw=1.2, color="gray")
            ax_prob.set_ylabel(prob_col)
            ax_prob.set_ylim(-0.05, 1.05)
            ax_prob.set_title(f"{label_name.capitalize()} probability")
            format_time_axis(ax_prob)

            ax.set_xlabel("Time")
            fig.suptitle(f"{region_name} - {sensorid} - {label_name.capitalize()}", fontsize=14)
            fig.tight_layout()
            file_region_name = "_".join(region_name.lower().split(" "))
            save_path = SAVE_PATH / label_name / f"{file_region_name}_{sensorid}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=200)
            plt.close(fig=fig)


if __name__ == "__main__":
    main()
