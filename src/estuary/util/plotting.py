from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_estuaries_ca(
    gdf: gpd.GeoDataFrame,
    column: str | None = None,
    categorical: bool | None = None,
    *,
    cmap: str | plt.Colormap = "viridis",
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
    - If `column` is provided, plots points colored by that column.
      *Numeric columns* use a colorbar (default).
      *Categorical columns* (strings/ints/classes) use a discrete colormap + legend.
      You may force categorical behavior via `categorical=True`.

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
        data_raw = gdf[column]

        # Decide whether this is numeric or categorical.
        # - If categorical is explicitly set, obey it.
        # - Otherwise, attempt numeric coercion and treat as numeric only if most non-null values
        #       parse.
        non_null = data_raw.notna()
        if categorical is None:
            data_num = pd.to_numeric(data_raw, errors="coerce")
            denom = int(non_null.sum())
            frac_parsed = (int(data_num.notna().sum()) / denom) if denom > 0 else 0.0
            is_categorical = frac_parsed < 0.95
        else:
            is_categorical = bool(categorical)
            data_num = pd.to_numeric(data_raw, errors="coerce") if not is_categorical else None  # type: ignore

        # --------------------------------------------------------------
        # Categorical mode: discrete colors + legend
        # --------------------------------------------------------------
        if is_categorical:
            # Treat NaNs as invalid
            valid_mask = data_raw.notna().to_numpy()

            # Categories can be strings or ints; keep stable ordering
            cats = pd.Index(data_raw[non_null].astype(object).unique()).tolist()
            # Try to sort for nicer legends; fall back to original order if not sortable
            try:
                cats = sorted(cats)
            except Exception:
                pass

            cat_to_code: dict[Any, int] = {c: i for i, c in enumerate(cats)}
            codes = np.full(len(data_raw), -1, dtype=int)
            for i, v in enumerate(data_raw.astype(object).to_numpy()):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                codes[i] = cat_to_code.get(v, -1)

            ncat = max(len(cats), 1)
            if isinstance(cmap, plt.Colormap):
                cmap_obj = cmap
            else:
                cmap_obj = plt.get_cmap(cmap, ncat)
            norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, ncat + 0.5, 1), ncolors=ncat)

            # Plot invalid values in gray
            if (~valid_mask).any():
                ax.scatter(
                    x[~valid_mask],
                    y[~valid_mask],
                    s=markersize,
                    color="lightgray",
                    alpha=point_alpha,
                    edgecolor="black",
                    linewidth=0.4,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )

            # Plot valid categorical values
            sc = ax.scatter(
                x[valid_mask],
                y[valid_mask],
                s=markersize,
                c=codes[valid_mask],
                cmap=cmap_obj,
                norm=norm,
                alpha=point_alpha,
                edgecolor="black",
                linewidth=0.4,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

            # Legend (one handle per category)
            handles = []
            labels = []
            for c in cats:
                code = cat_to_code[c]
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markersize=markersize * 0.25,
                        markerfacecolor=cmap_obj(code),
                        markeredgecolor="black",
                        markeredgewidth=0.4,
                    )
                )
                labels.append(str(c))

            ax.legend(handles, labels, title=column, loc="upper right", frameon=True)

        # --------------------------------------------------------------
        # Numeric mode: continuous colors + colorbar (existing behavior)
        # --------------------------------------------------------------
        else:
            assert data_num is not None
            data = data_num.astype(float)
            valid = np.isfinite(data)

            if vmin is None:
                vmin = float(np.nanmin(data))
            if vmax is None:
                vmax = float(np.nanmax(data))

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
            fig.savefig(save_path, dpi=300, bbox_inches="tight")  # type: ignore

        if show:
            plt.show()
        else:
            plt.close(fig)  # type: ignore
