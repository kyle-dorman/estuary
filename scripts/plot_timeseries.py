#!/usr/bin/env python3
"""
plot_region_states.py

Read a CSV of predictions with columns: acquired, region, pred, conf
and writes one PNG per region showing open/closed over time plus a
rolling window summary curve.

Usage:
  python scripts/plot_region_states.py \
      --preds /path/to/preds.csv \
      --out-dir /path/to/output_dir \
      --class0 "Closed" --class1 "Open" \
      --window-days 7 --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from matplotlib.dates import ConciseDateFormatter, MonthLocator

from estuary.clay.data import parse_dt_from_pth


def add_acquired(df):
    # Prefer provided 'acquired' column; else derive from source_tif if available
    if "acquired" in df.columns:
        df["acquired"] = pd.to_datetime(df["acquired"], errors="coerce")
    elif "source_tif" in df.columns:
        df["acquired"] = df.source_tif.apply(lambda a: parse_dt_from_pth(Path(a)))
        df["acquired"] = pd.to_datetime(df["acquired"], errors="coerce")
    else:
        raise SystemExit("preds.csv must contain 'acquired' or 'source_tif' to derive dates")

    df = df.dropna(subset=["acquired"]).sort_values(["region", "acquired"])
    return df


def make_plot(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    high_res_df: pd.DataFrame,
    region: str,
    out_path: Path,
    class0: str,
    class1: str,
    dpi: int,
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
    conf_smooth_days: int = 21,
    skip_labels: bool = False,
) -> None:
    g = df[df["region"] == region].copy()
    if g.empty:
        return

    # Ensure proper types
    g["acquired"] = pd.to_datetime(g["acquired"], errors="coerce")
    g = g.dropna(subset=["acquired"]).sort_values("acquired")

    # Index by time
    g = g.set_index("acquired")

    # Optional date filtering
    if start_dt is not None or end_dt is not None:
        g = g.loc[start_dt:end_dt]
        if g.empty:
            return

    # Colors (1=open -> green, 0=closed -> red)
    cmap = {1: ("#2ca02c", class0), 0: ("#d62728", class1)}
    colors = [cmap[int(v)][0] for v in g["pred"].tolist()]

    # Build figure with two rows: predictions (top) and labels (bottom)
    nrows = 1 if skip_labels else 2
    width = 10 if not skip_labels else 4
    height = 3 if not skip_labels else 1.5
    fig, axes = plt.subplots(
        nrows=nrows,
        sharex=True,
        figsize=(width, height),
        dpi=dpi,
        constrained_layout=True,
        # gridspec_kw={"height_ratios": [2, 1]},
    )
    if skip_labels:
        ax = axes
    else:
        ax = axes[0]

    # Scatter of predicted states at y={0,1}
    ax.scatter(
        g.index,
        g["pred"].to_numpy(),
        c=colors,
        s=10,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.1,
        zorder=3,
    )

    # Confidence as a smoothed line on a twin y-axis (0..1)
    if "conf" in g.columns:
        conf = 1 - pd.to_numeric(g["conf"], errors="coerce").clip(0, 1)
        conf_smooth = conf.rolling(f"{conf_smooth_days}D", min_periods=1, center=True).mean()
        ax2 = ax.twinx()
        ax2.plot(
            g.index,
            conf_smooth.to_numpy(),
            color="#4c78a8",
            alpha=0.8,
            linewidth=1.8,
            zorder=1,
        )
        ax2.set_ylim(0.0, 1.0)
        ax2.set_yticks([0.0, 0.5, 1.0])
        ax2.set_ylabel("confidence", fontsize=8)
        ax2.grid(False)

    # Formatting for predictions axis
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([class0, class1])
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Predictions", fontsize=10)

    if not skip_labels:
        ax_lab = axes[1]
        # Labels subplot
        lab = labels_df[labels_df["region"] == region].copy()
        if not lab.empty:
            lab["acquired"] = pd.to_datetime(lab["acquired"], errors="coerce")
            lab = lab.dropna(subset=["acquired"]).sort_values("acquired").set_index("acquired")
            if start_dt is not None or end_dt is not None:
                lab = lab.loc[start_dt:end_dt]
            if not lab.empty:
                colors_lab = [cmap[int(v)][0] for v in lab["label_idx"].tolist()]
                ax_lab.scatter(
                    lab.index,
                    lab["label_idx"].to_numpy(),
                    c=colors_lab,
                    s=10,
                    alpha=0.9,
                    edgecolors="k",
                    linewidths=0.1,
                    zorder=2,
                )
        # Overlay markers for dates where we have high-resolution example images
        hi = high_res_df[high_res_df["region"] == region].copy()
        if not hi.empty:
            hi["acquired"] = pd.to_datetime(hi["acquired"], errors="coerce")
            hi = hi.dropna(subset=["acquired"]).sort_values("acquired").set_index("acquired")
            if start_dt is not None or end_dt is not None:
                hi = hi.loc[start_dt:end_dt]
            if not hi.empty:
                # Light vertical lines and a small marker below the 0/1 band
                ax_lab.vlines(
                    hi.index,
                    ymin=-0.2,
                    ymax=1.2,
                    colors="#4c78a8",
                    alpha=0.4,
                    linewidth=1.0,
                    zorder=1,
                )
                ax_lab.scatter(
                    hi.index,
                    [-0.15] * len(hi),
                    s=18,
                    color="#4c78a8",
                    alpha=0.9,
                    marker="v",
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )

        ax_lab.set_ylim(-0.25, 1.25)
        ax_lab.set_yticks([0, 1])
        ax_lab.set_yticklabels([class0, class1])
        ax_lab.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax_lab.set_title("Labels", fontsize=10)

    if skip_labels:
        label_axis = ax
    else:
        label_axis = axes[1]
        # Hide top x tick labels to reduce clutter
        ax.tick_params(labelbottom=False)

    # Quarterly month ticks (Jan, Apr, Jul, Oct) on bottom axis for seasonal sense
    q_locator = MonthLocator()
    label_axis.xaxis.set_major_locator(q_locator)
    label_axis.xaxis.set_major_formatter(ConciseDateFormatter(q_locator))
    label_axis.xaxis.set_minor_locator(MonthLocator())
    label_axis.tick_params(axis="x", which="minor", length=3, color="#999999", width=0.6)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot per-region open/closed time series from preds.csv"
    )
    ap.add_argument(
        "--preds",
        type=Path,
        required=True,
        help="Path to preds.csv with acquired, region, pred, conf",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels.csv with acquired, region, label",
    )
    ap.add_argument(
        "--high-res",
        type=Path,
        required=True,
        help="Path to directory of high res images",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("plots_region_states"), help="Directory to write PNGs"
    )
    ap.add_argument("--class0", type=str, default="Closed", help="Label for class 0")
    ap.add_argument("--class1", type=str, default="Open", help="Label for class 1")
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI")
    ap.add_argument("--start", type=str, default=None, help="Optional start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="Optional end date (YYYY-MM-DD)")
    ap.add_argument(
        "--conf-smooth-days",
        type=int,
        default=10,
        help="Rolling window (days) for confidence smoothing",
    )
    ap.add_argument("--region", type=str, default=None, help="If set, only plot this region")
    ap.add_argument("--skip-labels", action="store_true", help="If set, skip plotting labels")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    labels_df["label_idx"] = labels_df.label.apply(lambda a: int(a == "open"))
    labels_df = labels_df[labels_df.label.isin(["open", "closed"])].copy(deep=True).reset_index()

    df = pd.read_csv(args.preds)
    thresholds = pd.read_csv(Path(args.preds).parent / "thresholds.csv")
    for _, row in thresholds.iterrows():
        k = df.region == row.region
        df.loc[k, "pred"] = 1 - (df[k].conf >= row.best_threshold).astype(int)
    df = add_acquired(df)
    labels_df = add_acquired(labels_df)

    high_res = []
    for pth in Path(args.high_res).glob("*/*/files/*_pansharpened_clip.tif"):
        yearmonthday = pth.stem.split("_")[0]
        dt = pd.to_datetime(yearmonthday, format="%Y%m%d")
        high_res.append([pth, pth.parent.parent.name, dt])
    high_res_df = pd.DataFrame(high_res, columns=["path", "region", "acquired"])

    start_dt = pd.to_datetime(args.start) if args.start else None
    end_dt = pd.to_datetime(args.end) if args.end else None

    # Iterate regions and save one PNG each
    regions = df["region"].dropna().unique().tolist()
    if args.region:
        if args.region in regions:
            regions = [args.region]
        else:
            print(f"Region {args.region} not found in predictions file")
            return
    for region in tqdm.tqdm(regions):
        fname = out_dir / f"{region}_state_timeseries.png"
        make_plot(
            df,
            labels_df,
            high_res_df,
            region,
            fname,
            args.class0,
            args.class1,
            args.dpi,
            start_dt,
            end_dt,
            args.conf_smooth_days,
            args.skip_labels,
        )


if __name__ == "__main__":
    main()
