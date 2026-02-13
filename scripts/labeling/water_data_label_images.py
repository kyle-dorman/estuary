from pathlib import Path

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from PIL import Image

from estuary.util.img import tif_to_rgb


def filter_outliers(df: pd.DataFrame, depth_col: str = "height") -> pd.DataFrame:
    # Quick-and-dirty outlier removal on depth values using IQR
    y = df[depth_col].astype(float)
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1

    # Keep points within 3 * IQR (lenient to preserve events)
    lo = q1 - 3 * iqr
    hi = q3 + 3 * iqr

    mask = (y >= lo) & (y <= hi)
    return df.loc[mask].copy()


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

        out.append(filter_outliers(seg.copy()))

    return out


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
    "-i",
    "--image-base-path",
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to all images.",
)
@click.option(
    "-r",
    "--regions-path",
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to regions info.",
)
@click.option(
    "-w",
    "--water-data-path",
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to parsed water data.",
)
@click.option(
    "-tl",
    "--to-label-path",
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to list ok image to label.",
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
    image_base_path: Path,
    regions_path: Path,
    water_data_path: Path,
    to_label_path: Path,
    save_dir: Path,
    min_segment_length: int,
):
    regions_gdf = gpd.read_file(regions_path).rename(columns={"Site code": "region"})
    regions_gdf = regions_gdf[~regions_gdf.skipped].copy()
    regions_gdf = regions_gdf.set_index("region")

    to_label = pd.read_csv(to_label_path)
    to_label["acquired"] = pd.to_datetime(to_label["acquired"], errors="coerce", utc=True)
    to_label = to_label.sort_values(by=["region", "acquired"])
    to_label = to_label[(~to_label["acquired"].isna())].copy()

    water_to_label = []

    for region, _ in tqdm.tqdm(regions_gdf.iterrows(), total=len(regions_gdf)):
        region_water_data_dir = water_data_path / str(region)
        if not region_water_data_dir.exists():
            continue

        region_segments = []
        for p in region_water_data_dir.glob("*.csv"):
            df = pd.read_csv(p, dtype={"sensor_id": "string", "source": "string"})
            for _, sdf in df.groupby(["sensor_id", "source"]):
                segments = contiguous_segments(sdf, time_col="timestamp_utc")
                region_segments.extend(segments)

        if len(region_segments) == 0:
            continue

        region_segments = sorted(
            region_segments,
            key=lambda sdf: sdf.timestamp_utc.max() - sdf.timestamp_utc.min(),
            reverse=True,
        )

        region_to_label = to_label[to_label.region == region]
        for _, row in region_to_label.iterrows():
            acquired: pd.Timestamp = row["acquired"]
            asset_id = row["asset_id"]
            year = row["year"]
            month = row["month"]
            instrument = row["instrument"]
            delta = pd.Timedelta(days=min_segment_length // 2)
            start_dt = acquired - delta
            end_dt = acquired + delta

            # No month for skysat
            if instrument == "skysat":
                base = image_base_path / instrument / "results" / str(year) / str(region) / "files"
                source_tif = next(
                    (p for p in base.glob(f"{asset_id}*_pansharpened_clip.tif")), None
                )
            else:
                base = (
                    image_base_path
                    / instrument
                    / "results"
                    / str(year)
                    / str(month)
                    / str(region)
                    / "files"
                )
                source_tif = next((p for p in base.glob(f"{asset_id}*SR*clip.tif")), None)

            if source_tif is None:
                print("WARNING: Missing tif", instrument, year, month, region)
                continue

            l_segments = [
                s
                for s in region_segments
                if (s.timestamp_utc.min() < start_dt and end_dt < s.timestamp_utc.max())
            ]
            if not len(l_segments):
                continue

            segment = l_segments[0]
            disp_segment = segment[
                (segment.timestamp_utc > start_dt) & (segment.timestamp_utc < end_dt)
            ].set_index("timestamp_utc")
            fig_path = save_dir / "plots" / str(region) / f"{source_tif.stem}.jpeg"
            fig_path.parent.mkdir(exist_ok=True, parents=True)
            plot_depth_centered_on_event(disp_segment, "height", acquired, fig_path)

            img_path = save_dir / "images" / str(region) / f"{source_tif.stem}.jpeg"
            img_path.parent.mkdir(exist_ok=True, parents=True)
            tif_to_jpeg(source_tif, img_path)

            water_to_label.append(
                {
                    "region": region,
                    "source_tif": source_tif,
                    "seg_source": segment.source.iloc[0],
                    "img_path": img_path,
                    "fig_path": fig_path,
                }
            )

    pd.DataFrame(water_to_label).to_csv(save_dir / "to_label.csv", index=False)

    print("Done!")


if __name__ == "__main__":
    main()
