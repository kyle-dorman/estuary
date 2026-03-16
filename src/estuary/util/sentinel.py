import numpy as np
import rasterio

from estuary.util.img import contrast_stretch


def s2_broad_band(bands):
    blue = np.log10(1 + bands[10:12].mean(axis=0))
    green = np.log10(1 + bands[0:3].mean(axis=0))
    red = np.log10(1 + bands[3:9].mean(axis=0))

    img = np.dstack([red, green, blue])
    return contrast_stretch(img.transpose((2, 0, 1))).transpose((1, 2, 0))


def s2_tri_stimulus(bands):
    red_recipe = np.log10(
        1.0
        + 0.01 * bands[0]
        + 0.09 * bands[1]
        + 0.35 * bands[2]
        + 0.04 * bands[3]
        + 0.01 * bands[4]
        + 0.59 * bands[5]
        + 0.85 * bands[6]
        + 0.12 * bands[7]
        + 0.07 * bands[9]
        + 0.04 * bands[10]
    )
    green_recipe = np.log10(
        1.0
        + 0.26 * bands[2]
        + 0.21 * bands[3]
        + 0.50 * bands[4]
        + 1.00 * bands[5]
        + 0.38 * bands[6]
        + 0.04 * bands[7]
        + 0.03 * bands[9]
        + 0.02 * bands[10]
    )
    blue_recipe = np.log10(
        1.0
        + 0.07 * bands[0]
        + 0.28 * bands[1]
        + 1.77 * bands[2]
        + 0.47 * bands[3]
        + 0.16 * bands[4]
    )

    rgb_tri = np.dstack((red_recipe, green_recipe, blue_recipe))
    mn = rgb_tri.min()
    rgb_tri -= mn
    rgb_tri /= rgb_tri.max()
    return contrast_stretch(rgb_tri.transpose((2, 0, 1))).transpose((1, 2, 0))


S2_BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]


def s2_img(s2_dir):
    bands = []
    for name in S2_BAND_NAMES:
        file = list(s2_dir.glob(f"*_{name}_*"))[0]
        with rasterio.open(file) as src:
            bands.append(src.read(1))

    return s2_broad_band(np.array(bands))
