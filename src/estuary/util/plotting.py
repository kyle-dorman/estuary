from collections.abc import Sequence
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def plot_estuaries_ca(
    gdf: gpd.GeoDataFrame,
    column: str | None = None,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    scale: str = "linear",  # "linear" or "log"
    bins: Sequence[float] | None = None,
    figsize: tuple[int, int] = (10, 6),
    markersize: float = 50,
    point_color: str = "#d62728",
    point_alpha: float = 0.9,
    jitter_deg: float = 0.0,
    random_seed: int = 0,
    show_labels: bool = False,
    label_column: str | None = None,
    label_fontsize: int = 7,
    title: str | None = None,
    title_fontsize: int = 16,
    add_colorbar: bool = True,
    save_path: Path | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
):
    """Plot California estuaries on a Cartopy map.

    - If `column is None`, plots **locations only** (all points same color).
    - If `column` is provided, plots points colored by that numeric column.

    Notes on close/overlapping sites
    -------------------------------
    If some estuaries are extremely close in space, points may overlap.
    Use `jitter_deg` (e.g., 0.02–0.05) to add a small random offset in degrees,
    or increase `markersize`/use labels to help distinguish.

    Assumptions
    -----------
    - gdf is in EPSG:4326
    - geometries are POINTS

    If `ax` is provided, the function will draw on that axis and will NOT call
    `plt.show()` or save the figure. This allows composing multi-panel figures
    with shared colorbars handled externally.
    """

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError("GeoDataFrame must be in EPSG:4326")

    if column is not None and column not in gdf.columns:
        raise KeyError(f"{column} not found in GeoDataFrame")

    # Compute padded extent around California estuaries
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = max(0.5, (maxx - minx) * 0.15)
    pad_y = max(0.5, (maxy - miny) * 0.15)
    extent = (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)

    # Coordinates (optionally jittered)
    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
    if jitter_deg and jitter_deg > 0:
        rng = np.random.default_rng(random_seed)
        x = x + rng.uniform(-jitter_deg, jitter_deg, size=len(x))
        y = y + rng.uniform(-jitter_deg, jitter_deg, size=len(y))

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        created_ax = True
    else:
        fig = ax.figure
        created_ax = False

    ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore

    # Background
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dbe9ff", zorder=0)  # type: ignore
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f6f6f4", zorder=1)  # type: ignore
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, zorder=2)  # type: ignore
    ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.4, zorder=2)  # type: ignore

    # ------------------------------------------------------------------
    # Locations-only mode
    # ------------------------------------------------------------------
    if column is None:
        ax.scatter(
            x,
            y,
            s=markersize,
            color=point_color,
            alpha=point_alpha,
            edgecolor="black",
            linewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    # ------------------------------------------------------------------
    # Colored-by-column mode
    # ------------------------------------------------------------------
    else:
        data = gdf[column].astype(float)
        valid = np.isfinite(data)

        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)

        # Color normalization
        if bins is not None:
            norm = colors.BoundaryNorm(bins, ncolors=len(bins) - 1)
            cmap_obj = plt.get_cmap(cmap, len(bins) - 1)
        elif scale == "log":
            assert vmin is not None
            norm = colors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)

        # Plot invalid values in gray
        if (~valid).any():
            ax.scatter(
                x[~valid.to_numpy()],  # type: ignore
                y[~valid.to_numpy()],  # type: ignore
                s=markersize,
                color="lightgray",
                alpha=point_alpha,
                edgecolor="black",
                linewidth=0.4,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        # Plot valid values
        sc = ax.scatter(
            x[valid.to_numpy()],  # type: ignore
            y[valid.to_numpy()],  # type: ignore
            s=markersize,
            c=data[valid],
            cmap=cmap_obj,
            norm=norm,
            alpha=point_alpha,
            edgecolor="black",
            linewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        # Colorbar
        if add_colorbar:
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label(column)

    # Labels (optional)
    if show_labels:
        if label_column is not None and label_column in gdf.columns:
            labels = gdf[label_column].astype(str).to_numpy()
        else:
            labels = gdf.index.astype(str).to_numpy()

        for xi, yi, lab in zip(x, y, labels, strict=True):
            ax.text(
                float(xi),
                float(yi),
                str(lab),
                fontsize=label_fontsize,
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=12)

    if created_ax:
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
