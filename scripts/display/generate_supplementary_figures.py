#!/usr/bin/env python3
"""Generate publication-sized supplementary manuscript figures.

The analysis notebook remains the provenance entrypoint and passes its prepared
data objects to these functions. Keeping the drawing code here makes final-size
typography, dimensions, filenames, and validation reproducible.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import confusion_matrix

from estuary.util.data import parse_dt_from_pth
from estuary.util.img import false_color
from scripts.display.generate_submission_figures import (
    FULL_WIDTH_MM,
    add_panel_label,
    inches,
    plot_false_color,
    publication_styled,
)
from scripts.display.plot_water_timeseries import format_daily_date_axis

FIGURE_S1_SOURCES = (
    Path(
        "/Users/kyledorman/data/estuary/dataset/images/dove/results/2017/7/2147/files/"
        "20170723_181219_1007_3B_AnalyticMS_SR_clip.tif"
    ),
    Path(
        "/Users/kyledorman/data/estuary/dataset/images/dove/results/2018/6/43/files/"
        "20180609_181605_1021_3B_AnalyticMS_SR_clip.tif"
    ),
    Path(
        "/Users/kyledorman/data/estuary/dataset/images/dove/results/2022/3/83/files/"
        "20220302_181207_0e20_3B_AnalyticMS_SR_clip.tif"
    ),
    Path(
        "/Users/kyledorman/data/estuary/dataset/images/dove/results/2018/12/81/files/"
        "20181203_183436_1001_3B_AnalyticMS_SR_clip.tif"
    ),
)

FIGURE_S6_SOURCES = (
    Path(
        "/Users/kyledorman/data/estuary/dataset/images/dove/results/2019/7/13009/files/"
        "20190701_173312_1048_3B_AnalyticMS_SR_clip.tif"
    ),
    Path(
        "/Users/kyledorman/data/estuary/dataset/images/superdove/results/2025/2/2147/files/"
        "20250205_191554_60_24f2_3B_AnalyticMS_SR_8b_clip.tif"
    ),
    Path(
        "/Users/kyledorman/data/estuary/dataset/images/superdove/results/2024/2/51/files/"
        "20240206_180833_10_24a1_3B_AnalyticMS_SR_8b_clip.tif"
    ),
    Path(
        "/Users/kyledorman/data/estuary/dataset/images/dove/results/2021/4/12097/files/"
        "20210401_154810_1049_3B_AnalyticMS_SR_clip.tif"
    ),
)


def _output(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _read_false_color(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.read(1, masked=True).mask
    return false_color(data, nodata)


def _add_fixed_square_panel_label(ax: plt.Axes, label: str) -> None:
    """Center a panel letter in a fixed-size black square."""
    left = 0.015
    bottom = 0.93
    size = 0.055
    ax.add_patch(
        Rectangle(
            (left, bottom),
            size,
            size,
            transform=ax.transAxes,
            facecolor="black",
            edgecolor="none",
            alpha=0.65,
            zorder=9,
        )
    )
    ax.text(
        left + size / 2,
        bottom + size / 2,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        zorder=10,
    )


@publication_styled
def figure_s1(output_dir: Path) -> Path:
    """Low-quality examples as a hybrid, full-width PDF."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(inches(FULL_WIDTH_MM), inches(FULL_WIDTH_MM)),
        layout="constrained",
    )
    for ax, path, label in zip(
        axes.flat,
        FIGURE_S1_SOURCES,
        ("A", "B", "C", "D"),
        strict=True,
    ):
        plot_false_color(path, ax)
        add_panel_label(ax, label, image_panel=True)

    output_path = _output(output_dir, "figS01_low_quality_examples.pdf")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _matrix(labels: list[str], truth: pd.Series, prediction: pd.Series) -> np.ndarray:
    truth_index = np.array([labels.index(label) for label in truth])
    prediction_index = np.array([labels.index(label) for label in prediction])
    return confusion_matrix(truth_index, prediction_index, labels=[0, 1, 2])


@publication_styled
def figure_s2(
    d_ss_pairs: pd.DataFrame,
    dd_pairs: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Combine the former S2 and S3 confusion matrices as panels A and B."""
    labels = ["closed", "perched open", "open"]
    display_labels = ["Closed", "Partial", "Open"]
    matrices = (
        _matrix(labels, d_ss_pairs["label_ss"], d_ss_pairs["label_dd"]),
        _matrix(labels, dd_pairs["label_water"], dd_pairs["label_img"]),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(inches(FULL_WIDTH_MM), inches(82)),
        layout="constrained",
    )
    panel_specs = (
        (
            "A",
            "Cross-sensor label agreement",
            "PlanetScope label",
            "SkySat label",
        ),
        (
            "B",
            "Water-level-assisted label agreement",
            "Image-only label",
            "Image + water-level label",
        ),
    )
    for ax, matrix, (panel, title, xlabel, ylabel) in zip(
        axes,
        matrices,
        panel_specs,
        strict=True,
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar=False,
            square=True,
            annot_kws={"fontsize": 8},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        add_panel_label(ax, panel, image_panel=True)

    output_path = _output(output_dir, "figS02_label_agreement_confusion_matrices.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


@publication_styled
def figure_s3(d_ss_pairs: pd.DataFrame, output_dir: Path) -> Path:
    """SkySat and PlanetScope disagreement example with platform overlays."""
    disagreement = d_ss_pairs[
        (d_ss_pairs["label_ss"] == "open") & (d_ss_pairs["label_dd"] == "perched open")
    ]
    if len(disagreement) < 2:
        raise ValueError("Expected at least two SkySat/PlanetScope disagreement examples.")
    row = disagreement.iloc[1]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(inches(FULL_WIDTH_MM), inches(95)),
        layout="constrained",
    )
    for ax, path, label in (
        (axes[0], Path(row["source_tif_ss"]), "A  SkySat"),
        (axes[1], Path(row["source_tif_dd"]), "B  PlanetScope"),
    ):
        plot_false_color(path, ax)
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox={
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 1.5,
            },
        )

    output_path = _output(output_dir, "figS03_skysat_planetscope_disagreement.pdf")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


@publication_styled
def figure_s4(
    skysat_image: np.ndarray,
    planetscope_image: np.ndarray,
    sentinel2_image: np.ndarray,
    output_dir: Path,
) -> Path:
    """Spatial-resolution comparison with enlarged panel lettering."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(inches(FULL_WIDTH_MM), inches(65)),
        layout="constrained",
    )
    for ax, image, label in zip(
        axes,
        (skysat_image, planetscope_image, sentinel2_image),
        ("A", "B", "C"),
        strict=True,
    ):
        ax.imshow(image, interpolation="none", resample=False)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox={
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 1.5,
            },
        )

    output_path = _output(output_dir, "figS04_spatial_resolution_comparison.pdf")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


@publication_styled
def figure_s5(
    gap: pd.Series,
    comparison: pd.DataFrame,
    rho: float,
    p_value: float,
    output_dir: Path,
) -> Path:
    """Generalization-gap panels in a compact side-by-side layout."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(inches(FULL_WIDTH_MM), inches(76)),
        layout="constrained",
    )
    sns.histplot(
        gap,
        bins=12,
        color="0.35",
        edgecolor="white",
        linewidth=0.8,
        ax=axes[0],
    )
    axes[0].axvline(
        gap.median(),
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    axes[0].set_xlabel("F1 generalization gap (seen - unseen)")
    axes[0].set_ylabel("Number of estuaries")
    axes[0].text(
        0.98,
        0.95,
        f"Median: {gap.median():.02f}\nMaximum: {gap.max():.02f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )
    sns.regplot(
        data=comparison,
        x="f1_gap",
        y="entropy_gap",
        scatter_kws={"s": 20, "alpha": 0.8, "color": "#4C72B0"},
        line_kws={"color": "black", "linewidth": 1.2},
        ci=95,
        ax=axes[1],
    )
    axes[1].set_xlabel("F1 generalization gap (seen - unseen)")
    axes[1].set_ylabel("Entropy increase (unseen - seen)")
    axes[1].text(
        0.98,
        0.05,
        f"Spearman ρ = {rho:.2f}\np = {p_value:.3g}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
    )
    axes[1].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4,
                color="#4C72B0",
                label="Estuary",
            ),
            Line2D([], [], color="black", linewidth=1.2, label="Linear fit"),
            Patch(facecolor="0.7", edgecolor="none", alpha=0.5, label="95% CI"),
        ],
        loc="upper left",
        frameon=False,
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_path = _output(output_dir, "figS05_generalization_gap_entropy.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


@publication_styled
def figure_s6(output_dir: Path) -> Path:
    """Model-challenge examples as a hybrid, full-width PDF."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(inches(FULL_WIDTH_MM), inches(FULL_WIDTH_MM)),
        layout="constrained",
    )
    panels = (
        (axes[0, 0], FIGURE_S6_SOURCES[0], "A"),
        (axes[1, 0], FIGURE_S6_SOURCES[1], "B"),
        (axes[0, 1], FIGURE_S6_SOURCES[2], "C"),
        (axes[1, 1], FIGURE_S6_SOURCES[3], "D"),
    )
    for ax, path, label in panels:
        plot_false_color(path, ax)
        _add_fixed_square_panel_label(ax, label)

    output_path = _output(output_dir, "figS06_model_challenges.pdf")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_region_example(
    full: pd.DataFrame,
    all_segments: dict[int, list[pd.DataFrame]],
    gdf: pd.DataFrame,
    region: int,
    date: pd.Timestamp,
    axes: tuple[plt.Axes, plt.Axes] | np.ndarray,
) -> tuple[list, list[str]]:
    date = pd.Timestamp(date)
    selected = full.loc[region, date]
    source_tif = Path(selected.source_tif)
    capture_time = pd.Timestamp(parse_dt_from_pth(source_tif), tz="UTC")
    display_start = capture_time - pd.Timedelta("4D4h")
    display_end = capture_time + pd.Timedelta("4D4h")

    predictions = full.reset_index()
    predictions["acquired"] = pd.to_datetime(
        predictions["acquired"],
        errors="coerce",
        utc=True,
    )
    predictions = predictions[
        (predictions.region == region) & predictions.acquired.between(display_start, display_end)
    ].sort_values("acquired")

    image_ax = axes[0]
    ax = axes[1]
    water_ax = ax.twinx()
    observed = predictions[predictions.source_tif.notna()]
    ax.scatter(
        observed["acquired"],
        observed["p_open"],
        color="#0072B2",
        s=16,
        alpha=0.9,
        label="Open probability",
        zorder=4,
    )

    for segment in all_segments[region]:
        data = segment.copy()
        data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
        data = data.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        data = (
            data.set_index("timestamp_utc")
            .resample("1h", origin="epoch")
            .median(numeric_only=True)
            .dropna()
            .reset_index()
        )
        data = data[
            data["timestamp_utc"].between(
                display_start,
                display_end,
                inclusive="left",
            )
        ]
        if data.empty:
            continue
        time = data["timestamp_utc"]
        water_ax.plot(
            time,
            data["height"],
            color="#56B4E9",
            linewidth=1.2,
            label="Water level",
        )
        water_ax.plot(
            time,
            data["tide"],
            color="#CC79A7",
            linewidth=1.1,
            linestyle="--",
            label="Tide height",
        )
        ax.plot(
            time,
            data["tidal_attenuation"].astype(float),
            color="#8B1A1A",
            linewidth=1.2,
            linestyle="-.",
            label="Tidal attenuation",
        )

    ax.axvline(capture_time, linestyle=":", color="black", linewidth=1.2)
    ax.set_xlim(display_start, display_end)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel("Probability / tidal attenuation")
    water_ax.set_ylabel("Water level (m)")
    format_daily_date_axis(ax, display_start, display_end)

    plot_false_color(source_tif, image_ax)
    name = gdf.loc[region]["Estuary_Name"]
    image_ax.set_title(f"{name} - {capture_time:%Y-%m-%d}")

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = water_ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles_1 + handles_2, labels_1 + labels_2, strict=True):
        unique.setdefault(label, handle)
    return list(unique.values()), list(unique.keys())


def _example_figure(
    full: pd.DataFrame,
    all_segments: dict[int, list[pd.DataFrame]],
    gdf: pd.DataFrame,
    selections: tuple[tuple[int, pd.Timestamp], ...],
    output_dir: Path,
    filename: str,
    height_mm: float,
) -> Path:
    fig, axes = plt.subplots(
        len(selections),
        2,
        figsize=(inches(FULL_WIDTH_MM), inches(height_mm)),
        gridspec_kw={"width_ratios": [1, 2]},
        layout="constrained",
    )
    legend_handles: list = []
    legend_labels: list[str] = []
    for row, (region, date) in enumerate(selections):
        handles, labels = _plot_region_example(
            full,
            all_segments,
            gdf,
            region,
            date,
            axes[row],
        )
        if row == 0:
            legend_handles, legend_labels = handles, labels
        add_panel_label(
            axes[row, 0],
            chr(ord("A") + row),
            image_panel=True,
        )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="outside upper center",
        ncols=4,
        frameon=False,
    )
    output_path = _output(output_dir, filename)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


@publication_styled
def figure_s7(
    full: pd.DataFrame,
    all_segments: dict[int, list[pd.DataFrame]],
    gdf: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Former S8 closed-with-tidal-signal examples with daily ticks."""
    selections = (
        (43, pd.Timestamp(datetime(2024, 5, 8))),
        (59, pd.Timestamp(datetime(2018, 2, 25))),
        (48, pd.Timestamp(datetime(2020, 2, 15))),
    )
    return _example_figure(
        full,
        all_segments,
        gdf,
        selections,
        output_dir,
        "figS07_closed_with_tidal_signal_examples.pdf",
        180,
    )


@publication_styled
def figure_s8(
    full: pd.DataFrame,
    all_segments: dict[int, list[pd.DataFrame]],
    gdf: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Former S9 examples, excluding San Pedro Creek, with daily ticks."""
    selections = (
        (48, pd.Timestamp(datetime(2018, 2, 3))),
        (72, pd.Timestamp(datetime(2023, 10, 16))),
        (11, pd.Timestamp(datetime(2022, 5, 5))),
        (50, pd.Timestamp(datetime(2022, 2, 17))),
    )
    return _example_figure(
        full,
        all_segments,
        gdf,
        selections,
        output_dir,
        "figS08_open_with_low_tidal_signal_examples.pdf",
        225,
    )
