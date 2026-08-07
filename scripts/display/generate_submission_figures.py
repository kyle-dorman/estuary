#!/usr/bin/env python3
"""Generate manuscript Figures 2, 4, 5, 6, and 7 at final print size.

Run from the repository root with::

    uv run python -m scripts.display.generate_submission_figures
"""

from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from estuary.util.data import parse_dt_from_pth
from estuary.util.img import false_color
from estuary.util.plotting import plot_estuaries_ca
from scripts.display.plot_water_timeseries import (
    DEFAULT_WATER_DATA_PATH,
    plot_water_timeseries_for_tif,
)
from scripts.display.plot_water_timeseries import (
    save_plot as save_water_plot,
)

MM_PER_INCH = 25.4
FULL_WIDTH_MM = 190
ONE_COLUMN_WIDTH_MM = 90

BASE = Path("/Volumes/x10pro/estuary")
PREDICTIONS_PATH = Path(
    "/Volumes/x10pro/results/estuary/train/20260305-095554/merged_all_regions_timeseries_preds.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/kyledorman/data/estuary/display/final_paper_figures/official_submission"
)
DEFAULT_WATER_PLOT_DIR = Path(
    "/Users/kyledorman/data/estuary/display/final_paper_figures/water_plots"
)

PUBLICATION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.title_fontsize": 7.5,
    "axes.grid": False,
    "xtick.bottom": True,
    "xtick.major.size": 3,
    "xtick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.left": True,
    "ytick.major.size": 3,
    "ytick.major.width": 0.6,
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}


def publication_styled(function):
    """Run a figure generator with publication typography in any call order."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with mpl.rc_context(PUBLICATION_STYLE):
            return function(*args, **kwargs)

    return wrapped


FIGURE_2_SOURCES = {
    "A": Path(
        "/Users/kyledorman/data/estuary/dataset/images/dove/results/2017/8/11/files/"
        "20170801_174611_1010_3B_AnalyticMS_SR_clip.tif"
    ),
    "B": Path(
        "/Users/kyledorman/data/estuary/dataset/images/superdove/results/2021/12/21/files/"
        "20211217_183631_29_2402_3B_AnalyticMS_SR_8b_clip.tif"
    ),
    "C": Path(
        "/Users/kyledorman/data/estuary/dataset/images/dove/results/2019/6/34/files/"
        "20190607_182323_1035_3B_AnalyticMS_SR_clip.tif"
    ),
}


def inches(mm: float) -> float:
    """Convert millimeters to inches for Matplotlib."""
    return mm / MM_PER_INCH


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    image_panel: bool = False,
) -> None:
    """Add a consistent, final-size panel label."""
    kwargs = {}
    if image_panel:
        kwargs = {
            "color": "white",
            "bbox": {
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 1.5,
            },
        }
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        zorder=10,
        **kwargs,
    )


def plot_false_color(path: Path, ax: plt.Axes) -> None:
    """Draw a native-resolution false-color image without raster resampling."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.read(1, masked=True).mask
    ax.imshow(false_color(data, nodata), interpolation="none", resample=False)
    ax.axis("off")


def load_analysis_data() -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the current estuary inventory, predictions, and daily probabilities."""
    gdf = gpd.read_file(BASE / "geos/ca_data_w_empa_pmep_usgs.geojson")
    gdf = gdf.rename(columns={"Site code": "region"})
    gdf = gdf[~gdf.skipped].copy().set_index("region")

    gdf_points = gdf.copy()
    gdf_points.geometry = gdf.to_crs("ESRI:54008").geometry.centroid.to_crs(gdf.crs)

    preds_all = pd.read_csv(PREDICTIONS_PATH)
    preds_all["acquired"] = pd.to_datetime(preds_all["acquired"], errors="coerce", utc=True)
    preds_all["date"] = pd.to_datetime(preds_all["acquired"].dt.date)
    preds_all = (
        preds_all[preds_all.region.isin(gdf.index) & (preds_all.y_pred_unsure != 1)]
        .copy()
        .sort_values(["region", "acquired"])
        .reset_index(drop=True)
    )

    start = datetime(year=2018, month=1, day=1)
    full_dates = pd.date_range(start, preds_all["date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [preds_all["region"].unique(), full_dates],
        names=["region", "date"],
    )
    full = preds_all.set_index(["region", "date"]).sort_index().reindex(full_index)

    for key in ("p_closed", "p_open"):
        full[key] = (
            full[key]
            .groupby(level="region")
            .transform(
                lambda series: series.interpolate(
                    method="linear",
                    limit_direction="both",
                )
            )
        )

    probabilities = full[["p_closed", "p_open"]].to_numpy(dtype=float)
    row_sum = np.nansum(probabilities, axis=1, keepdims=True)
    valid = np.isfinite(row_sum[:, 0]) & (row_sum[:, 0] > 0)
    probabilities[valid] = probabilities[valid] / row_sum[valid]
    full.loc[:, ["p_closed", "p_open"]] = probabilities

    region_min_date = preds_all.groupby("region")["date"].min().max()
    region_max_date = preds_all.groupby("region")["date"].max().min()
    full = full.reset_index()
    full = full[full.date.between(region_min_date, region_max_date)].copy()
    full = full.set_index(["region", "date"])
    return gdf, gdf_points, preds_all, full


@publication_styled
def figure_2(output_dir: Path, water_plot_dir: Path) -> Path:
    """Generate Figure 2 with matched, vector water-level plots."""
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(inches(FULL_WIDTH_MM), inches(FULL_WIDTH_MM)),
        gridspec_kw={"width_ratios": [1, 2]},
        layout="constrained",
    )

    for row, (label, tif_path) in enumerate(FIGURE_2_SOURCES.items()):
        image_ax, water_ax = axes[row]
        plot_false_color(tif_path, image_ax)
        add_panel_label(image_ax, label, image_panel=True)

        water = plot_water_timeseries_for_tif(
            tif_path,
            water_ax,
            water_data_path=DEFAULT_WATER_DATA_PATH,
        )

        acquired = pd.Timestamp(parse_dt_from_pth(tif_path), tz="UTC")
        refreshed_path = water_plot_dir / str(int(tif_path.parent.parent.name))
        refreshed_path = refreshed_path / f"{tif_path.stem}.jpeg"
        save_water_plot(water, acquired, refreshed_path, dpi=300)

    output_path = output_dir / "fig02_connectivity_state_examples.pdf"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


@publication_styled
def figure_4(
    gdf_points: gpd.GeoDataFrame,
    preds_all: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Generate the full-width spatial-coverage and temporal-gap maps."""
    start = datetime(year=2018, month=1, day=1, tzinfo=UTC)
    end = preds_all["acquired"].max().normalize()
    n_days_total = len(pd.date_range(start, end, freq="D"))
    observations = preds_all[preds_all.acquired > start].copy()
    days_per_site = (
        observations.assign(date=observations["acquired"].dt.normalize())
        .groupby("region")["date"]
        .nunique()
    )

    valid_label = "Days with valid observations (%)"
    gap_label = "Maximum observation gap (days)"
    valid_days = (100 * days_per_site / n_days_total).rename(valid_label)

    def max_gap_days(times: pd.Series) -> float:
        gaps = times.sort_values().diff().dt.days.dropna()
        return float(gaps.max()) if len(gaps) else 0.0

    maximum_gap = observations.groupby("region")["acquired"].apply(max_gap_days).rename(gap_label)
    coverage = gdf_points[["geometry"]].join(valid_days, how="left")
    gaps = gdf_points[["geometry"]].join(maximum_gap, how="left")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(inches(FULL_WIDTH_MM), inches(94)),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    for ax, data, column, label in zip(
        axes,
        (coverage, gaps),
        (valid_label, gap_label),
        ("A", "B"),
        strict=True,
    ):
        plot_estuaries_ca(
            data,
            column=column,
            cmap="viridis",
            add_colorbar=True,
            show=False,
            markersize=20,
            jitter_deg=0.01,
            point_alpha=0.8,
            ax=ax,
        )
        add_panel_label(ax, label)

    output_path = output_dir / "fig04_spatial_coverage_and_gaps.pdf"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_connectivity_timeseries(
    full: pd.DataFrame,
    region: int,
    ax: plt.Axes,
    *,
    legend: bool = False,
) -> None:
    """Draw the seven-day rolling open/closed probabilities for one estuary."""
    data = full.loc[region].sort_index()
    rolling = data[["p_open", "p_closed"]].rolling("7D", min_periods=1).mean()
    ax.plot(rolling.index, rolling["p_open"], label="Open probability", lw=1.1)
    ax.plot(rolling.index, rolling["p_closed"], label="Closed probability", lw=1.1)

    dominant = rolling.idxmax(axis=1)
    for column, color, label in (
        ("p_open", "tab:blue", "Open state"),
        ("p_closed", "tab:orange", "Closed state"),
    ):
        ax.fill_between(
            rolling.index,
            0,
            1,
            where=dominant == column,
            color=color,
            alpha=0.08,
            label=label,
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("7-day mean probability")
    ax.set_xlabel("Date")
    if legend:
        ax.legend(loc="lower left")


@publication_styled
def figure_5(
    gdf: gpd.GeoDataFrame,
    preds_all: pd.DataFrame,
    full: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Generate the full-width example connectivity time-series figure."""
    selections = (
        ("A", 2145, preds_all[(preds_all.region == 2145) & (preds_all.y_pred == 1)].iloc[100]),
        ("B", 51, preds_all[(preds_all.region == 51) & (preds_all.y_pred == 1)].iloc[0]),
        (
            "C",
            35,
            preds_all[
                (preds_all.region == 35) & (preds_all.y_pred == 1) & (preds_all.p_open < 0.9)
            ].iloc[1],
        ),
    )
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(inches(FULL_WIDTH_MM), inches(193)),
        gridspec_kw={"width_ratios": [1, 2]},
        layout="constrained",
    )

    for row, (label, region, record) in enumerate(selections):
        image_ax, series_ax = axes[row]
        plot_false_color(Path(record.source_tif), image_ax)
        image_ax.set_title(str(gdf.loc[region, "Estuary_Name"]))
        add_panel_label(image_ax, label, image_panel=True)
        plot_connectivity_timeseries(
            full,
            region,
            series_ax,
            legend=row == 0,
        )

    output_path = output_dir / "fig05_example_connectivity_time_series.pdf"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


@publication_styled
def figure_6(
    gdf_points: gpd.GeoDataFrame,
    full: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Generate the one-column map of mean predicted open probability."""
    label = "Mean predicted open probability"
    mean_open = full["p_open"].groupby(level="region").mean().rename(label)
    mapped = gdf_points.join(mean_open, how="left")

    fig, ax = plt.subplots(
        figsize=(inches(ONE_COLUMN_WIDTH_MM), inches(92)),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    plot_estuaries_ca(
        mapped,
        ax=ax,
        column=label,
        cmap="viridis",
        markersize=20,
        jitter_deg=0.05,
        point_alpha=0.8,
        add_colorbar=True,
        show=False,
    )

    output_path = output_dir / "fig06_state_map_p_open.pdf"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


@publication_styled
def figure_7(full: pd.DataFrame, output_dir: Path) -> Path:
    """Generate the full-width monthly median and IQR openness figure."""
    data = full.reset_index().copy()
    data["date"] = pd.to_datetime(data["date"])
    data["wy_month"] = (data["date"].dt.month - 10) % 12
    region_month = data.groupby(["region", "wy_month"])["p_open"].mean().reset_index()
    quantiles = (
        region_month.groupby("wy_month")["p_open"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})
    )

    months = np.arange(12)
    month_labels = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]
    fig, ax = plt.subplots(
        figsize=(inches(FULL_WIDTH_MM), inches(61)),
        layout="constrained",
    )
    ax.plot(months, quantiles["median"], lw=1.5, color="tab:blue", label="Median")
    ax.fill_between(
        months,
        quantiles["q25"],
        quantiles["q75"],
        color="tab:blue",
        alpha=0.25,
        label="IQR",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(months, month_labels)
    ax.set_xlabel("Water year month (Oct-Sep)")
    ax.set_ylabel("Median predicted openness")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    output_path = output_dir / "fig07_statewide_seasonal_openness.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_all(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    water_plot_dir: Path = DEFAULT_WATER_PLOT_DIR,
) -> list[Path]:
    """Generate all approved revised figures and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    water_plot_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(PUBLICATION_STYLE):
        gdf, gdf_points, preds_all, full = load_analysis_data()
        return [
            figure_2(output_dir, water_plot_dir),
            figure_4(gdf_points, preds_all, output_dir),
            figure_5(gdf, preds_all, full, output_dir),
            figure_6(gdf_points, full, output_dir),
            figure_7(full, output_dir),
        ]


if __name__ == "__main__":
    for generated_path in generate_all():
        print(generated_path)
