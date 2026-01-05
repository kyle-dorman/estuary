from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def add_season_fields(dates: pd.DatetimeIndex, season_year_mode="djf_to_next_year"):
    """
    season: DJF/MAM/JJA/SON
    season_year_mode:
      - "djf_to_next_year": DJF is labeled by year of Jan/Feb (e.g., Dec 2020 -> season_year 2021)
      - "calendar": all seasons labeled by calendar year of the timestamp
    """
    m = dates.month
    season = np.where(
        m.isin([12, 1, 2]),
        "DJF",
        np.where(m.isin([3, 4, 5]), "MAM", np.where(m.isin([6, 7, 8]), "JJA", "SON")),
    )

    if season_year_mode == "calendar":
        season_year = dates.year
    elif season_year_mode == "djf_to_next_year":
        # Dec counts toward next year's DJF
        season_year = dates.year + (m == 12).astype(int)
    else:
        raise ValueError("season_year_mode must be 'djf_to_next_year' or 'calendar'")

    return season, season_year


def add_water_year_fields(dates: pd.DatetimeIndex, start_month=10, label="end"):
    """
    Water year (default Oct-Sep). If label='end', WY 2021 = Oct 2020 .. Sep 2021.
    """
    y = dates.year
    m = dates.month
    if label == "end":
        wy = y + (m >= start_month).astype(int)
    elif label == "start":
        wy = y - (m < start_month).astype(int)
    else:
        raise ValueError("label must be 'end' or 'start'")
    return wy


def aggregate_region_time(
    full: pd.DataFrame,
    value_col="y_prob",
    how="D",
    agg="mean",
    week_ends="SUN",
    season_year_mode="djf_to_next_year",
    water_year_start_month=10,
    water_year_label="end",
):
    """
    full: DataFrame indexed by ['region','date'] with a column value_col.
    how:
      - 'D' (daily / no aggregation)
      - 'W' (weekly)
      - 'M' (monthly)
      - 'Y' (calendar year)
      - 'WY' (water year)
      - 'S' (meteorological season: DJF/MAM/JJA/SON)
    Returns:
      mat: (n_regions, n_bins) float array
      regions: index labels for rows
      time_bins: labels for columns (PeriodIndex-ish or tuples)
      wide_df: DataFrame region x bin (for debugging / plotting)
    """
    df = full.reset_index()[["region", "date", value_col]].copy()
    df["date"] = pd.to_datetime(df["date"])

    if how == "D":
        # treat each day as a bin (still do a pivot)
        df["bin"] = df["date"]
    elif how == "W":
        # weekly bins, choose week ending day
        rule = f"W-{week_ends.upper()}"
        df["bin"] = df["date"].dt.to_period(rule)  # type: ignore
    elif how == "M":
        df["bin"] = df["date"].dt.to_period("M")  # type: ignore
    elif how == "Y":
        df["bin"] = df["date"].dt.to_period("Y")  # type: ignore
    elif how == "WY":
        wy = add_water_year_fields(
            df["date"].dt.to_pydatetime().__class__(df["date"]),  # type: ignore
            start_month=water_year_start_month,
            label=water_year_label,
        )
        # easier: vectorized via numpy on datetime64
        dates = pd.DatetimeIndex(df["date"].values)
        wy = add_water_year_fields(
            dates, start_month=water_year_start_month, label=water_year_label
        )
        df["bin"] = wy.astype(int)
    elif how == "S":
        dates = pd.DatetimeIndex(df["date"].values)
        season, season_year = add_season_fields(dates, season_year_mode=season_year_mode)
        df["bin"] = list(
            zip(season_year.astype(int), season, strict=True)
        )  # (year, 'DJF'/'MAM'...)
    else:
        raise ValueError("how must be one of: 'D','W','M','Y','WY','S'")

    # aggregate within each (region, bin)
    grouped = df.groupby(["region", "bin"], sort=True)[value_col].agg(agg).reset_index()

    # pivot to region x bin
    wide = (
        grouped.pivot(index="region", columns="bin", values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # (optional) drop bins that have any missing across regions, to keep pairwise comparability
    # strict wide = wide.dropna(axis=1, how="any")

    # fill remaining gaps (if any) per region across bins; for coarse bins you may prefer
    # interpolation
    wide = wide.apply(lambda s: s.interpolate(limit_direction="both").ffill().bfill(), axis=1)

    mat = wide.to_numpy(dtype=float)
    return mat, wide.index.to_numpy(), wide.columns, wide


def monthly_climatology_matrix(full: pd.DataFrame, value_col: str = "y_prob", agg: str = "mean"):
    """Return a (n_regions, 12) matrix where each column is month-of-year (1..12).

    This collapses inter-annual variability and focuses clustering on seasonality shape.
    """
    df = full.reset_index()[["region", "date", value_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["moy"] = df["date"].dt.month

    reg_moy = df.groupby(["region", "moy"], sort=True)[value_col].agg(agg).reset_index()

    wide = (
        reg_moy.pivot(index="region", columns="moy", values=value_col)
        .sort_index(axis=0)
        .reindex(columns=list(range(1, 13)))
    )

    # Fill any missing month-of-year for a region (rare) conservatively
    wide = wide.apply(lambda s: s.interpolate(limit_direction="both").ffill().bfill(), axis=1)

    Y = wide.to_numpy(dtype=float)
    regions = wide.index.to_numpy()
    months = wide.columns.to_numpy()
    return Y, regions, months, wide


# ------------- New helpers for monthly timeseries, static-site flagging, and lagged correlation -------------
def monthly_timeseries_matrix(full: pd.DataFrame, value_col: str = "y_prob", agg: str = "mean"):
    """Return (Y, regions, months, wide) where Y is region x (year,month) monthly mean.

    Columns are a PeriodIndex (year-month), preserving interannual variability.
    """
    df = full.reset_index()[["region", "date", value_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ym"] = df["date"].dt.to_period("M")

    reg_ym = df.groupby(["region", "ym"], sort=True)[value_col].agg(agg).reset_index()

    wide = (
        reg_ym.pivot(index="region", columns="ym", values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # Fill gaps conservatively
    wide = wide.apply(lambda s: s.interpolate(limit_direction="both").ffill().bfill(), axis=1)

    Y = wide.to_numpy(dtype=float)
    regions = wide.index.to_numpy()
    months = wide.columns
    return Y, regions, months, wide


def flag_static_sites(
    wide_monthly: pd.DataFrame,
    *,
    std_thresh: float = 0.05,
    extreme_frac: float = 0.95,
    lo: float = 0.1,
    hi: float = 0.9,
) -> pd.Series:
    """Flag sites with negligible temporal variability.

    A site is static if:
      - std(monthly y_prob) < std_thresh, OR
      - fraction of months <= lo or >= hi exceeds extreme_frac

    Returns Series indexed by region with values: 'dynamic', 'always_open', 'always_closed'.
    """
    out = {}
    for region, s in wide_monthly.iterrows():
        v = s.values.astype(float)
        sd = np.nanstd(v)
        frac_open = np.mean(v >= hi)
        frac_closed = np.mean(v <= lo)

        if sd < std_thresh:
            if frac_open >= frac_closed:
                out[region] = "always_open"
            else:
                out[region] = "always_closed"
        elif frac_open >= extreme_frac:
            out[region] = "always_open"
        elif frac_closed >= extreme_frac:
            out[region] = "always_closed"
        else:
            out[region] = "dynamic"

    return pd.Series(out, name="site_type")


def lagged_corr_similarity(Y: np.ndarray, max_lag: int = 2) -> np.ndarray:
    """Max Pearson correlation over lags [-max_lag, +max_lag] for each pair.

    Y: (n_regions, n_time) monthly anomaly matrix (zero-mean per row recommended)
    Returns similarity matrix S in [-1, 1].
    """
    n, T = Y.shape
    S = np.eye(n)

    for i in range(n):
        xi = Y[i]
        for j in range(i + 1, n):
            xj = Y[j]
            best = -np.inf
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    a, b = xi[:lag], xj[-lag:]
                elif lag > 0:
                    a, b = xi[lag:], xj[:-lag]
                else:
                    a, b = xi, xj

                if len(a) < 3:
                    continue
                r = np.corrcoef(a, b)[0, 1]
                if np.isfinite(r):
                    best = max(best, r)

            S[i, j] = S[j, i] = best if np.isfinite(best) else 0.0

    return S


def corr_similarity_matrix(Y, regions):
    """
    Y: (n_regions, n_time) float array
    returns DataFrame S where S[i,j] in [-1,1]
    """
    Y = np.asarray(Y, dtype=float)

    # handle constant rows (std=0) -> correlation undefined; set them to 0 signal
    row_std = Y.std(axis=1, keepdims=True)
    Yz = np.where(row_std == 0, 0.0, (Y - Y.mean(axis=1, keepdims=True)) / row_std)

    # Pearson corr = dot(z_i, z_j) / (T-1) (corrcoef does the bookkeeping)
    S = np.corrcoef(Yz)

    # numerical cleanup
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(S, index=regions, columns=regions)


def reorder_by_hclust(
    sim_df: pd.DataFrame,
    method: str = "average",
    *,
    sim_range: tuple[float, float] = (-1.0, 1.0),
    # 'one_minus' for similarity; 'one_minus_abs' if you want sign-invariant
    distance: str = "one_minus",
):
    """Reorder a similarity matrix using hierarchical clustering.

    Parameters
    ----------
    sim_df:
        Square similarity matrix (DataFrame). For correlation similarities,
            values are typically in [-1, 1].
    method:
        Linkage method (e.g., 'average', 'complete').
    sim_range:
        Range to clip similarities to before converting to distance.
    distance:
        How to convert similarity -> distance.
        - 'one_minus': D = 1 - S
        - 'one_minus_abs': D = 1 - |S|  (treats strong negative correlation as similar)
        - 'angular': D = sqrt(0.5 * (1 - S))  (correlation -> angular distance; keeps range in [0, 1])

    Returns
    -------
    clust_df, order, Z
    """
    S = sim_df.to_numpy(dtype=float)
    lo, hi = sim_range
    S = np.clip(S, lo, hi)

    if distance == "one_minus":
        D = 1.0 - S
    elif distance == "one_minus_abs":
        D = 1.0 - np.abs(S)
    elif distance == "angular":
        # Angular distance derived from correlation; bounded in [0, 1] for S in [-1, 1]
        # This often behaves better than (1 - corr) for hierarchical clustering.
        D = np.sqrt(0.5 * (1.0 - S))
    else:
        raise ValueError("distance must be 'one_minus', 'one_minus_abs', or 'angular'")

    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method=method)
    order = leaves_list(Z)
    clust = sim_df.iloc[order, :].iloc[:, order]  # type: ignore
    return clust, order, Z


def plot_heatmap(
    sim_df: pd.DataFrame,
    title: str = "",
    show_labels: bool = False,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "Similarity",
    savepath: Path | None = None,
    show: bool = True,
):
    M = sim_df.to_numpy()
    labels = sim_df.index.to_numpy()

    if vmin is None:
        vmin = float(np.nanmin(M))
    if vmax is None:
        vmax = float(np.nanmax(M))

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel("Region")
    ax.set_ylabel("Region")

    if show_labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def monthly_cluster_stats(full: pd.DataFrame, cluster_map: pd.Series, value_col="y_prob"):
    """
    full: indexed by ['region','date'] with column value_col
    cluster_map: pd.Series region->cluster (1..k)
    returns a tidy DataFrame:
      month, cluster, mean, std, n_regions
    """
    df = full.reset_index()[["region", "date", value_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()  # type: ignore # month start timestamp

    # attach cluster id
    df["cluster"] = df["region"].map(cluster_map)

    # monthly mean per region
    reg_month = (
        df.groupby(["cluster", "region", "month"], sort=True)[value_col]
        .mean()
        .rename("region_month_mean")
        .reset_index()
    )

    # aggregate across regions within each cluster-month
    out = (
        reg_month.groupby(["cluster", "month"], sort=True)["region_month_mean"]
        .agg(mean="mean", std="std", n_regions="count")
        .reset_index()
        .sort_values(["cluster", "month"])
    )
    return out


def make_cluster_color_map(cluster_map: pd.Series, cmap_name: str = "tab20") -> dict[int, str]:
    """Return a stable mapping from cluster id -> hex color."""
    cluster_ids = sorted(pd.unique(cluster_map.dropna().astype(int)))
    cmap = cm.get_cmap(cmap_name, max(len(cluster_ids), 1))
    out: dict[int, str] = {}
    for i, cid in enumerate(cluster_ids):
        out[int(cid)] = mcolors.to_hex(cmap(i))
    return out


def plot_cluster_monthly_mean_std(
    cl_month: pd.DataFrame,
    title: str = "Monthly open probability by cluster",
    *,
    color_map: dict[int, str] | None = None,
    save_dir: Path | None = None,
    show: bool = True,
):
    """Plot one figure per cluster: mean ± std over time.

    cl_month columns: cluster, month, mean, std, n_regions
    color_map: optional mapping cluster_id -> color (hex). If None, uses tab20.
    """
    if cl_month.empty:
        return

    # Build a color map if not provided
    if color_map is None:
        tmp = pd.Series(cl_month["cluster"].astype(int).unique())
        color_map = make_cluster_color_map(pd.Series(tmp.values, index=tmp.values))

    for cluster_id, g in cl_month.groupby("cluster"):
        cluster_id = int(cluster_id)
        g = g.sort_values("month")
        x = g["month"].to_numpy()
        y = g["mean"].to_numpy()
        s = g["std"].fillna(0.0).to_numpy()
        n = int(g["n_regions"].max())

        col = color_map.get(cluster_id, "#333333")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, label=f"Cluster {cluster_id} (n={n})")
        ax.fill_between(x, y - s, y + s, alpha=0.2)

        # Force line + fill to use the same cluster color
        for line in ax.lines:
            line.set_color(col)
        # Fill_between returns PolyCollection; set its facecolor
        if ax.collections:
            ax.collections[-1].set_facecolor(col)
            ax.collections[-1].set_edgecolor("none")

        ax.set_title(f"{title} — Cluster {cluster_id}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean y_prob (open probability)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            outpath = save_dir / f"cluster_{cluster_id:02d}_monthly_mean_std.png"
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)


# --------------------- Plot clusters on map ---------------------
def plot_clusters_map(
    regions_df: gpd.GeoDataFrame,
    cluster_map: pd.Series,
    *,
    title: str = "Clusters over California",
    show_labels: bool = False,
    dropna: bool = True,
    color_map: dict[int, str] | None = None,
    savepath: Path | None = None,
    show: bool = True,
):
    # Join cluster ids onto geometry
    cm = cluster_map.rename("cluster")
    g = regions_df.join(cm, how="left")

    if dropna:
        g = g.dropna(subset=["cluster"])

    if g.empty:
        raise ValueError("No sites to plot after joining regions_df with cluster_map")

    # Ensure integer clusters for categorical plotting
    g["cluster"] = g["cluster"].astype(int)

    # Ensure CRS; assume WGS84 if missing, then plot in lon/lat
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    gdf = g.to_crs("EPSG:4326")

    if color_map is None:
        color_map = make_cluster_color_map(cluster_map)

    gdf["color"] = gdf["cluster"].astype(int).map(color_map).fillna("#333333")

    # Compute a padded extent around points
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = max(0.5, (maxx - minx) * 0.15)
    pad_y = max(0.5, (maxy - miny) * 0.15)
    extent = (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)

    # --- Plot with Cartopy ---
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore

    # Ocean/Land background (Natural Earth, 10m)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dbe9ff", zorder=0)  # type: ignore
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f6f6f4", edgecolor="none", zorder=1)  # type: ignore
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"), edgecolor="#4a4a4a", linewidth=0.6, zorder=2
    )  # type: ignore
    ax.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="#e8f2ff",
        edgecolor="#8aa6c1",
        linewidth=0.3,
        zorder=2,
    )  # type: ignore

    ax.add_feature(
        cfeature.STATES.with_scale("10m"),
        edgecolor="#999999",
        linewidth=0.5,
        zorder=2,
    )  # type: ignore

    # Plot points on the Cartopy axis using explicit scatter for color control
    ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        s=40,
        c=gdf["color"].to_numpy(),
        edgecolors="black",
        linewidths=0.5,
        alpha=0.95,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Legend with stable colors
    handles = [
        Patch(facecolor=color_map[int(cid)], edgecolor="black", label=f"Cluster {int(cid)}")
        for cid in sorted(pd.unique(gdf["cluster"].astype(int)))
        if int(cid) in color_map
    ]
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=True)

    if show_labels:
        for idx, row in gdf.iterrows():
            try:
                x, y = row.geometry.x, row.geometry.y
                ax.text(x, y, str(idx), fontsize=6, transform=ccrs.PlateCarree())
            except Exception:
                continue

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def hclust_from_similarity(S_df, method="average"):
    S = S_df.to_numpy(dtype=float)
    D = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method=method)
    order = leaves_list(Z)
    return Z, order


def summarize_cluster_cut(Z, *, thresholds: list[float], label: str = ""):
    """Print a quick summary of cluster sizes for a set of distance thresholds."""
    for t in thresholds:
        ids = fcluster(Z, t=t, criterion="distance")
        vc = pd.Series(ids).value_counts()
        n = len(vc)
        n_single = int((vc == 1).sum())
        biggest = int(vc.max())
        msg = f"t={t:>5}: n_clusters={n:>3}  singletons={n_single:>3}  max_size={biggest:>2}"
        if label:
            msg = f"[{label}] " + msg
        print(msg)


def mean_within_cluster_similarity(sim_df: pd.DataFrame, cluster_map: pd.Series) -> pd.Series:
    """Mean off-diagonal similarity within each cluster."""
    out = {}
    for cid, members in cluster_map.groupby(cluster_map).groups.items():
        idx = list(members)
        if len(idx) <= 1:
            out[cid] = np.nan
            continue
        S = sim_df.loc[idx, idx].to_numpy(dtype=float)
        # off-diagonal mean
        m = (S.sum() - np.trace(S)) / (S.size - len(idx))
        out[cid] = float(m)
    return pd.Series(out).sort_index()


def main():
    SAVE_DIR = Path("/Volumes/x10pro/estuary/display/clustering")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    run_tag = f"{time_label}_dist{dist}" if "time_label" in locals() else "run"

    preds_all = pd.read_csv(
        "/Users/kyledorman/data/results/estuary/train/20251114-130229/merged_all_regions_timeseries_preds.csv"
    )
    skipped_regions = pd.read_csv("/Volumes/x10pro/estuary/geos/skipped_regions.csv")[
        "Site code"
    ].to_list()
    preds_all["acquired"] = (
        pd.to_datetime(preds_all["acquired"], errors="coerce", utc=True)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    preds_all["date"] = preds_all.acquired.dt.date  # type: ignore
    preds_all["month"] = preds_all.acquired.dt.month  # type: ignore
    preds_all["year"] = preds_all.acquired.dt.year  # type: ignore
    preds_all = (
        preds_all[
            (~preds_all.region.isin(skipped_regions))
            & (preds_all.y_pred_unsure != 1)
            & (preds_all.year >= 2018)
        ]
        .copy()
        .sort_values(by=["region", "acquired"])
        .reset_index(drop=True)
    )

    # 1) Make sure date is datetime64[ns]
    df = preds_all.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 2) Get global date range
    start = df["date"].min()
    end = df["date"].max()

    # 3) Build full MultiIndex: all regions × all dates
    regions = df["region"].unique()
    full_dates = pd.date_range(start, end, freq="D")

    full_index = pd.MultiIndex.from_product(
        [regions, full_dates],  # type: ignore
        names=["region", "date"],
    )

    # 4) Reindex and forward-fill within each region
    full = df.set_index(["region", "date"]).sort_index().reindex(full_index)

    # Forward-fill predictions within each region
    # full["y_pred_smooth"] = full["y_pred_smooth"].groupby(level="region").ffill()

    # Forward-fill predictions within each region
    full["y_prob"] = (
        full["y_prob"]
        .groupby(level="region")
        .apply(lambda s: s.interpolate(method="linear", limit_direction="both"))
        .ffill()
        .reset_index(level=0, drop=True)
    )

    region_min_date = df.groupby("region").date.min().max()
    region_max_date = df.groupby("region").date.max().min()

    full = full.reset_index()
    full = full[full.date.between(region_min_date, region_max_date)].copy()
    full = full.set_index(["region", "date"])

    # # 5) Get a region × time array for IoU
    # # Rows: region, Cols: date
    # y_mat = full["y_pred_smooth"].unstack("date").astype(int).to_numpy()

    # y_mat_prob = full["y_prob"].unstack("date").to_numpy()

    # region_labels = full["y_pred_smooth"].unstack("date").index.to_list()

    # date_axis = full["y_pred_smooth"].unstack("date").columns.to_list()

    # --- Choose clustering time resolution ---
    # 'YM' = year-month (highest resolution, more dimensions)
    # 'YS' = year-season (lower resolution, often more stable)
    CLUSTER_RES = "YM"  # change to "YS" if you want year/season

    if CLUSTER_RES == "YM":
        # Year-month time series (preserves interannual variability)
        Y_all, regions_all, ym_bins, wide_all = monthly_timeseries_matrix(
            full, value_col="y_prob", agg="mean"
        )
        time_label = "year-month"

        # Static-site filtering based on the same monthly series
        site_type = flag_static_sites(wide_all)
        print(site_type.value_counts())
        dynamic_regions = site_type[site_type == "dynamic"].index.to_numpy()

        wide_dyn = wide_all.loc[dynamic_regions]
        Y_dyn = wide_dyn.to_numpy(dtype=float)

        # Similarity: max correlation over small lags (±1 month)
        S = lagged_corr_similarity(Y_dyn, max_lag=1)
        corr_df = pd.DataFrame(S, index=dynamic_regions, columns=dynamic_regions)

    elif CLUSTER_RES == "YS":
        # Year-season time series using bins like (season_year, season)
        Y_all, regions_all, bins, wide_all = aggregate_region_time(
            full, value_col="y_prob", how="S", agg="mean", season_year_mode="djf_to_next_year"
        )
        time_label = "year-season"

        # Static-site filtering: use std over seasonal bins
        sd = pd.Series(np.nanstd(Y_all, axis=1), index=regions_all)
        site_type = pd.Series(
            np.where(sd < 0.05, "static", "dynamic"), index=regions_all, name="site_type"
        )
        print(site_type.value_counts())
        dynamic_regions = site_type[site_type == "dynamic"].index.to_numpy()

        wide_dyn = wide_all.loc[dynamic_regions]
        Y_dyn = wide_dyn.to_numpy(dtype=float)

        # Similarity: max correlation over small lags (±1 season)
        S = lagged_corr_similarity(Y_dyn, max_lag=0)
        corr_df = pd.DataFrame(S, index=dynamic_regions, columns=dynamic_regions)

    else:
        raise ValueError("CLUSTER_RES must be 'YM' or 'YS'")

    corr_ord, order, Z = reorder_by_hclust(
        corr_df,
        method="average",
        sim_range=(-1.0, 1.0),
        distance="angular",
    )

    plot_heatmap(
        corr_ord,
        title=f"Pairwise lagged correlation (dynamic sites, {time_label}, ±1 lag)",
        show_labels=False,
        vmin=-1,
        vmax=1,
        cbar_label="Pearson correlation",
        savepath=SAVE_DIR / f"heatmap_corr_{time_label}.png",
        show=True,
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(Z, labels=corr_df.index.to_list(), leaf_rotation=90, ax=ax)
    ax.set_title(f"Hierarchical clustering dendrogram (angular distance from corr; {time_label})")
    plt.tight_layout()
    dendro_path = SAVE_DIR / f"dendrogram_{time_label}.png"
    fig.savefig(dendro_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Quick sweep to choose a distance threshold.
    # For angular distance, thresholds are in ~[0, 1].
    summarize_cluster_cut(Z, thresholds=[0.30, 0.35, 0.40, 0.45, 0.5], label="angular")

    # cluster cut: distance threshold on angular distance (lower => smaller/tighter clusters)
    dist = 0.45
    cluster_ids = fcluster(Z, t=dist, criterion="distance")

    cluster_map = pd.Series(cluster_ids, index=corr_df.index, name="cluster")
    print(cluster_map.value_counts().sort_index())

    # Drop singleton clusters for downstream time-series plotting
    cluster_sizes = cluster_map.value_counts()
    keep_clusters = cluster_sizes[cluster_sizes >= 3].index
    cluster_map_nosingle = cluster_map[cluster_map.isin(keep_clusters)]

    print(
        f"""Keeping {len(cluster_map_nosingle)} / {len(cluster_map)} 
        dynamic sites for plots (dropped singleton clusters)"""
    )

    # Build a stable color map for clusters
    color_map = make_cluster_color_map(cluster_map_nosingle)

    # quick quality check: mean within-cluster correlation
    print("mean within-cluster corr:\n", mean_within_cluster_similarity(corr_df, cluster_map))

    print("Excluded non-dynamic sites:")
    print(site_type[site_type != "dynamic"].value_counts())

    # cl_month = monthly_cluster_stats(full, cluster_map, value_col="y_prob")
    cl_month = monthly_cluster_stats(full, cluster_map_nosingle, value_col="y_prob")

    run_dir = SAVE_DIR / f"{time_label}_dist{dist:.2f}"
    run_dir.mkdir(parents=True, exist_ok=True)

    plot_cluster_monthly_mean_std(
        cl_month,
        title=f"Monthly mean ± std of y_prob by correlation clusters (dist={dist})",
        color_map=color_map,
        save_dir=run_dir / "cluster_timeseries",
        show=True,
    )

    regions_df = gpd.read_file("/Volumes/x10pro/estuary/geos/ca_data_w_empa_pmep.geojson")
    regions_df.geometry = regions_df.geometry.centroid
    regions_df = regions_df.set_index("Site code")

    # Geographic view of clusters (only non-singleton clusters used in time-series plot)
    plot_clusters_map(
        regions_df,
        cluster_map_nosingle,
        title=f"Clusters over California ({time_label}, dist={dist}, min_cluster_size=3)",
        show_labels=False,
        color_map=color_map,
        savepath=run_dir / "clusters_map.png",
        show=True,
    )

    # Optional: also show excluded / non-dynamic sites if you want to sanity check
    static_sites = site_type[site_type != "dynamic"].index
    plot_clusters_map(
        regions_df,
        pd.Series(0, index=static_sites),
        title="Excluded static sites",
        savepath=run_dir / "excluded_static_sites.png",
        show=True,
    )


if __name__ == "__main__":
    main()
