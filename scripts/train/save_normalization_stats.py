import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import tqdm
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import PowerTransformer

from estuary.util.constants import BAND_COLORS, BAND_NAMES, EIGHT_TO_4

logger = logging.getLogger(__name__)


@click.command()
@click.option("-l", "--labels-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-s", "--save-path", type=click.Path(exists=False, path_type=Path), required=True)
@click.option("-m", "--max-raw-pixel-value", type=int, default=7e3)
@click.option("-c", "--min_count", type=int, default=10)
@click.option("--max-std", type=float, default=3.0)
@click.option("--power-scale", is_flag=True)
@click.option("--stride", type=int, default=10)
def main(
    labels_path: Path,
    save_path: Path,
    max_raw_pixel_value: int,
    min_count: int,
    max_std: float,
    power_scale: bool,
    stride: int,
):
    logger.info("Calculating normalization statistics")

    save_path.mkdir(exist_ok=True, parents=True)

    labels = pd.read_csv(labels_path)
    labels = labels[labels.label != "unsure"]

    rng = np.random.default_rng(seed=42)

    # Prepare counters
    num_bins = max_raw_pixel_value
    all_counts = np.zeros((8, num_bins), dtype=np.int64)
    bins = np.linspace(0, max_raw_pixel_value + 1, num=num_bins + 1)

    # Load all rasters and count every stride-th pixel. Skip pixels with no-data.
    for p in tqdm.tqdm(labels.source_tif):
        with rasterio.open(p) as src:
            data = src.read(masked=True)

        start = rng.integers(0, stride)
        chip = data[:, start::stride, start::stride].clip(0, max_raw_pixel_value)

        if len(data) == 8:
            bands = list(range(8))
        else:
            bands = EIGHT_TO_4

        for idx, band in enumerate(bands):
            channel_chip = chip[idx].compressed()
            counts, _ = np.histogram(channel_chip, bins=bins)
            all_counts[band] += counts

    # Save the counts so the remainder can be recomputed faster.
    np.save(save_path / "counts.npy", all_counts)
    np.save(save_path / "bins.npy", bins)

    # Normalize the counts by the minumum count value (or the min_count)
    all_counts = all_counts.astype(np.float64)
    for i in range(8):
        min_i = max(min_count, all_counts[i][np.where(all_counts[i] > 0)].min())
        all_counts[i] /= min_i

    # Convert counts back to ints
    all_counts = np.ceil(all_counts).astype(np.int32)

    # Save plots of the before/after normalization per channel
    sns.set_theme()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, squeeze=True, figsize=(12, 6))

    # Prepare to save the means and standard deviations
    means = np.zeros(8, dtype=np.float64)
    stds = np.zeros(8, dtype=np.float64)
    lambdas = np.zeros(8, dtype=np.float64)

    # Create a new array where each element is the average of two neighboring values
    bins = bins.astype(np.float64)
    mean_bins = (bins[:-1] + bins[1:]) / 2

    for idx in range(8):
        # Recreate the raw values and plot as a distribution
        x = np.repeat(mean_bins, all_counts[idx])
        sns.kdeplot(
            x,
            ax=ax_l,
            color=BAND_COLORS[idx].lower(),
            label=BAND_NAMES[idx],
            clip=(0, mean_bins[-1]),
        )

        if power_scale:
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            xt = pt.fit_transform(x.reshape(-1, 1))[:, 0]
            scale = pt.transform(np.array([[max_raw_pixel_value]]))[0, 0]
            xt = xt / scale

            lambdas[idx] = pt.lambdas_[0]
        else:
            xt = x / max_raw_pixel_value

        scaler = StandardScaler()
        xt = scaler.fit_transform(xt.reshape(-1, 1))[:, 0]  # type: ignore

        assert scaler.mean_ is not None
        assert scaler.scale_ is not None

        means[idx] = scaler.mean_[0]
        stds[idx] = scaler.scale_[0]

        # Plot the normalized values
        sns.kdeplot(
            xt,
            ax=ax_r,
            color=BAND_COLORS[idx].lower(),
            label=BAND_NAMES[idx],
            clip=(-max_std, max_std),
        )

    ax_l.set_xlabel("Pixel values")
    ax_l.legend()
    ax_l.set_title("Without Normalization")

    ax_r.set_xlabel("Pixel values")
    ax_r.legend()
    ax_r.set_title("With Normalization")

    # Save the distrubution figure
    fig.savefig(str(save_path / "distribution.png"))

    # Save the dataset statisitics
    stats = {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "lambdas": lambdas.tolist(),
        "power_scale": power_scale,
        "max_raw_pixel_value": max_raw_pixel_value,
        "max_std": max_std,
    }
    with open(save_path / "stats.json", "w") as f:
        json.dump(stats, f)

    logger.info("Done")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
