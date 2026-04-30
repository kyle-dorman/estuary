from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SeasonalDriverConfig:
    valid_regions: tuple[int, ...] = (
        96,
        12103,
        92,
        84,
        13057,
        2138,
        56,
        57,
        51,
        50,
        48,
        33,
        32,
        31,
        22,
        21,
        16,
        15,
        14,
        11,
    )

    open_threshold: float = 0.440
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"

    min_coverage_p_open: float = 0.80
    min_coverage_driver: float = 0.80
    flicker_window_days: int = 2

    predictions_path: str = (
        "/Users/kyledorman/data/results/estuary/train/20260305-095554/"
        "merged_all_regions_timeseries_preds.csv"
    )
    waves_path: str = "/Volumes/x10pro/estuary/drivers_data/cdip_waves.parquet"
    tide_path: str = "/Volumes/x10pro/estuary/drivers_data/fes2022_tides.parquet"
    streamflow_path: str = "/Volumes/x10pro/estuary/drivers_data/usgs_data_iv.parquet"

    nearest_cdip_path: str = "/Volumes/x10pro/estuary/drivers_data/geos/nearest_cdip_points.csv"
    cdip_points_meta_path: str = "/Volumes/x10pro/estuary/drivers_data/geos/cdip_points_meta.csv"

    output_dir: str = (
        "/Users/kyledorman/data/results/estuary/train/20260305-095554/seasonal_drivers"
    )


# =========================
# Generic helpers
# =========================


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def add_time_fields(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day

    month_to_season = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall",
        10: "fall",
        11: "fall",
    }
    out["season"] = out["month"].map(month_to_season)

    out["water_year"] = np.where(out["month"] >= 10, out["year"] + 1, out["year"])

    def month_to_wy_season(month: int) -> str:
        if month in [10, 11, 12]:
            return "OND"
        if month in [1, 2, 3]:
            return "JFM"
        if month in [4, 5, 6]:
            return "AMJ"
        return "JAS"

    out["wy_season"] = out["month"].map(month_to_wy_season)
    return out


def wrap_angle_degrees(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return ((s + 180.0) % 360.0) - 180.0


def filter_cdip_wave_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["wave_flag_primary"] = pd.to_numeric(out["wave_flag_primary"], errors="coerce")
    out = out[out["wave_flag_primary"].isin([1, 2, 3])].copy()
    return out


def filter_usgs_iv_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["variable_code"] = out["variable_code"].astype(str)
    out = out[out["variable_code"] == "00060"].copy()

    if "approval_status" in out.columns:
        approval = out["approval_status"].astype(str).str.strip().str.lower()
        out = out[approval == "approved"].copy()
    else:
        qualifier = out.get("qualifier_codes", pd.Series("", index=out.index))
        qualifier = qualifier.fillna("").astype(str).str.upper()
        has_a = qualifier.str.contains(r"(?:^|\b|,|;)A(?:\b|,|;|$)", regex=True)
        has_p = qualifier.str.contains(r"(?:^|\b|,|;)P(?:\b|,|;|$)", regex=True)
        out = out[has_a & ~has_p].copy()

    return out


# =========================
# Predictions
# =========================


def normalize_probability_columns(
    df: pd.DataFrame,
    prob_cols: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    p = out[list(prob_cols)].to_numpy(dtype=float)
    row_sum = np.nansum(p, axis=1, keepdims=True)
    ok = np.isfinite(row_sum[:, 0]) & (row_sum[:, 0] > 0)

    p_norm = p.copy()
    p_norm[ok] = p[ok] / row_sum[ok]
    out.loc[:, list(prob_cols)] = p_norm
    return out


def prediction_label(y_pred_open: int) -> str:
    return "open" if y_pred_open == 1 else "closed"


def mark_online_prediction_flickers(
    df: pd.DataFrame,
    flicker_window_days: int,
    state_col: str = "y_pred_open_original",
) -> pd.DataFrame:
    out = df.copy()
    out["is_flicker_removed"] = False
    out["flicker_pattern"] = pd.NA
    out["flicker_previous_observation_date"] = pd.NaT
    out["flicker_next_observation_date"] = pd.NaT
    out["flicker_days_since_previous_observation"] = np.nan
    out["flicker_days_until_next_observation"] = np.nan

    for _, region_df in out.sort_values(["region", "date"]).groupby("region", sort=False):
        kept_indices: list[int] = []

        for idx in region_df.index:
            state = out.at[idx, state_col]
            if pd.isna(state):
                continue

            kept_indices.append(idx)

            while len(kept_indices) >= 3:
                prev_idx, mid_idx, next_idx = kept_indices[-3:]
                prev_state = int(out.at[prev_idx, state_col])
                mid_state = int(out.at[mid_idx, state_col])
                next_state = int(out.at[next_idx, state_col])

                prev_date = out.at[prev_idx, "date"]
                mid_date = out.at[mid_idx, "date"]
                next_date = out.at[next_idx, "date"]
                prev_gap_days = (mid_date - prev_date).total_seconds() / 86400.0
                next_gap_days = (next_date - mid_date).total_seconds() / 86400.0

                is_isolated_flip = prev_state == next_state and mid_state != prev_state
                within_window = (
                    0 <= prev_gap_days <= flicker_window_days
                    and 0 <= next_gap_days <= flicker_window_days
                )
                if not (is_isolated_flip and within_window):
                    break

                out.at[mid_idx, "is_flicker_removed"] = True
                out.at[mid_idx, "flicker_previous_observation_date"] = prev_date
                out.at[mid_idx, "flicker_next_observation_date"] = next_date
                out.at[mid_idx, "flicker_days_since_previous_observation"] = prev_gap_days
                out.at[mid_idx, "flicker_days_until_next_observation"] = next_gap_days
                out.at[mid_idx, "flicker_pattern"] = "/".join(
                    [
                        prediction_label(prev_state),
                        prediction_label(mid_state),
                        prediction_label(next_state),
                    ]
                )

                kept_indices.pop(-2)

    return out


def build_daily_p_open_table(
    preds_all: pd.DataFrame,
    valid_regions: Sequence[int],
    open_threshold: float,
    start_date: str,
    end_date: str | None = None,
    flicker_window_days: int = 2,
) -> pd.DataFrame:
    df = preds_all.copy()
    df = df[df["region"].isin(valid_regions)].copy()

    df["date"] = pd.to_datetime(df["date"])
    if end_date is None:
        end_date = str(df["date"].max().date())

    regions = sorted(df["region"].unique())
    full_dates = pd.date_range(start_date, end_date, freq="D")

    full_index = pd.MultiIndex.from_product(
        [regions, full_dates],
        names=["region", "date"],
    )

    prob_cols = ["p_closed", "p_open"]
    original_prob_cols = ["p_closed_original", "p_open_original"]
    for col in prob_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_original"] = df[col]

    df["y_pred_open_original"] = pd.Series(
        np.where(df["p_open"].notna(), (df["p_open"] > open_threshold).astype(int), pd.NA),
        index=df.index,
        dtype="Int64",
    )
    df = mark_online_prediction_flickers(
        df,
        flicker_window_days=flicker_window_days,
        state_col="y_pred_open_original",
    )
    df.loc[df["is_flicker_removed"], prob_cols] = np.nan

    full = df.set_index(["region", "date"]).sort_index().reindex(full_index)

    full["is_obs_original"] = full[original_prob_cols].notna().any(axis=1)
    full["is_obs"] = full[prob_cols].notna().any(axis=1)
    full["is_flicker_removed"] = full["is_flicker_removed"].fillna(False).astype(bool)

    for col in [*prob_cols, *original_prob_cols]:
        full[col] = (
            full[col]
            .groupby(level="region")
            .apply(lambda s: s.interpolate(method="linear", limit_direction="both"))
            .ffill()
            .bfill()
            .reset_index(level=0, drop=True)
        )

    full = normalize_probability_columns(full, prob_cols)
    full = normalize_probability_columns(full, original_prob_cols)

    region_min_date = df.groupby("region")["date"].min().max()
    region_max_date = df.groupby("region")["date"].max().min()

    full = full.reset_index()
    full = full[full["date"].between(region_min_date, region_max_date)].copy()

    real_dates = full.loc[full["is_obs"], ["region", "date"]].copy()
    previous_real = pd.merge_asof(
        full.sort_values(["date", "region"]),
        real_dates.rename(columns={"date": "previous_real_observation_date"}).sort_values(
            ["previous_real_observation_date", "region"]
        ),
        left_on="date",
        right_on="previous_real_observation_date",
        by="region",
        direction="backward",
    )
    next_real = pd.merge_asof(
        full.sort_values(["date", "region"]),
        real_dates.rename(columns={"date": "next_real_observation_date"}).sort_values(
            ["next_real_observation_date", "region"]
        ),
        left_on="date",
        right_on="next_real_observation_date",
        by="region",
        direction="forward",
    )
    full = previous_real.merge(
        next_real[["region", "date", "next_real_observation_date"]],
        on=["region", "date"],
        how="left",
    ).sort_values(["region", "date"])

    full["days_since_previous_real_observation"] = (
        full["date"] - full["previous_real_observation_date"]
    ).dt.days
    full["days_until_next_real_observation"] = (
        full["next_real_observation_date"] - full["date"]
    ).dt.days
    distance_cols = [
        "days_since_previous_real_observation",
        "days_until_next_real_observation",
    ]
    full["min_days_since_real_observation"] = full[distance_cols].min(axis=1)
    full["max_days_since_real_observation"] = full[distance_cols].max(axis=1)
    full["is_interpolated"] = ~full["is_obs"]

    full["y_pred_open"] = (full["p_open"] > open_threshold).astype(int)
    full["y_pred_open_original"] = (full["p_open_original"] > open_threshold).astype(int)

    eps = 1e-12
    probs = full[prob_cols].to_numpy(dtype=float)
    probs_clip = np.clip(probs, eps, 1.0)

    entropy = -np.sum(probs_clip * np.log(probs_clip), axis=1)
    entropy_norm = entropy / np.log(len(prob_cols))

    full["entropy"] = entropy
    full["entropy_norm"] = entropy_norm

    full = add_time_fields(full, date_col="date")
    return full


# =========================
# Wave prep
# =========================


def prepare_wave_orientation_lookup(
    nearest_cdip_df: pd.DataFrame,
    cdip_points_meta_df: pd.DataFrame,
) -> pd.DataFrame:
    nearest = nearest_cdip_df[["site_id", "cdip_id"]].copy().drop_duplicates()
    meta = cdip_points_meta_df[["cdip_id", "shore_normal"]].copy().drop_duplicates()
    meta["shore_normal"] = pd.to_numeric(meta["shore_normal"], errors="coerce")

    out = nearest.merge(meta, on="cdip_id", how="left")
    out = out.rename(columns={"site_id": "region"})
    out = out.dropna(subset=["region", "shore_normal"]).copy()
    out = out.drop_duplicates(subset=["region", "cdip_id"])
    return out[["region", "shore_normal"]]


def prepare_waves_for_daily(
    waves_df: pd.DataFrame,
    wave_orientation_lookup: pd.DataFrame,
    valid_regions: Sequence[int],
) -> pd.DataFrame:
    out = waves_df.copy()
    out = filter_cdip_wave_qc(out)

    out = out.rename(columns={"site_id": "region"})
    out = out[out["region"].isin(valid_regions)].copy()

    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True).dt.tz_localize(None)
    out["wave_hs"] = pd.to_numeric(out["wave_hs"], errors="coerce")
    out["wave_tp"] = pd.to_numeric(out["wave_tp"], errors="coerce")
    out["wave_dp"] = pd.to_numeric(out["wave_dp"], errors="coerce")

    out = out.merge(wave_orientation_lookup, on="region", how="left")
    out["shore_normal"] = pd.to_numeric(out["shore_normal"], errors="coerce")

    out = out.dropna(
        subset=["region", "time", "wave_hs", "wave_tp", "wave_dp", "shore_normal"]
    ).copy()
    out = out[(out["wave_hs"] >= 0) & (out["wave_tp"] >= 0)].copy()

    out["wave_forcing"] = (out["wave_hs"] ** 2) * out["wave_tp"]

    rel_deg = wrap_angle_degrees(out["wave_dp"] - out["shore_normal"])
    rel_rad = np.deg2rad(rel_deg)

    out["wave_forcing_cross"] = out["wave_forcing"] * np.abs(np.cos(rel_rad))
    out["wave_forcing_along"] = out["wave_forcing"] * np.abs(np.sin(rel_rad))

    out = out.sort_values(["region", "time"]).reset_index(drop=True)
    return out


def build_daily_wave_table(
    waves: pd.DataFrame,
    valid_regions: Sequence[int],
) -> pd.DataFrame:
    out = waves.copy()
    out = out[out["region"].isin(valid_regions)].copy()

    out["time"] = pd.to_datetime(out["time"])
    out["date"] = out["time"].dt.floor("D")

    numeric_cols = [
        "wave_hs",
        "wave_tp",
        "wave_forcing",
        "wave_forcing_cross",
        "wave_forcing_along",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    daily = out.groupby(["region", "date"], as_index=False).agg(
        wave_hs_mean=("wave_hs", "mean"),
        wave_hs_p90=("wave_hs", lambda s: s.quantile(0.9)),
        wave_hs_max=("wave_hs", "max"),
        wave_tp_mean=("wave_tp", "mean"),
        wave_tp_p90=("wave_tp", lambda s: s.quantile(0.9)),
        wave_tp_max=("wave_tp", "max"),
        wave_energy_mean=("wave_forcing", "mean"),
        wave_energy_p90=("wave_forcing", lambda s: s.quantile(0.9)),
        wave_energy_max=("wave_forcing", "max"),
        wave_energy_cross_mean=("wave_forcing_cross", "mean"),
        wave_energy_cross_p90=("wave_forcing_cross", lambda s: s.quantile(0.9)),
        wave_energy_cross_max=("wave_forcing_cross", "max"),
        wave_energy_along_mean=("wave_forcing_along", "mean"),
        wave_energy_along_p90=("wave_forcing_along", lambda s: s.quantile(0.9)),
        wave_energy_along_max=("wave_forcing_along", "max"),
        wave_n_obs=("wave_hs", lambda s: s.notna().sum()),
    )

    daily = add_time_fields(daily, date_col="date")
    return daily


def zero_aware_rank_norm(s):
    out = pd.Series(np.nan, index=s.index, dtype=float)

    is_zero = s == 0
    is_pos = s > 0

    out[is_zero] = 0.0
    out[is_pos] = s[is_pos].rank(pct=True)

    return out


def add_per_region_rank_normalization(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_rank_norm"] = out.groupby("region")[col].transform(zero_aware_rank_norm)
    return out


DRIVER_RANK_COLUMNS = [
    "wave_hs_mean",
    "wave_hs_p90",
    "wave_hs_max",
    "wave_tp_mean",
    "wave_tp_p90",
    "wave_tp_max",
    "wave_energy_mean",
    "wave_energy_p90",
    "wave_energy_max",
    "wave_energy_cross_mean",
    "wave_energy_cross_p90",
    "wave_energy_cross_max",
    "wave_energy_along_mean",
    "wave_energy_along_p90",
    "wave_energy_along_max",
    "streamflow_mean",
    "streamflow_median",
    "streamflow_p90",
    "streamflow_max",
    "tide_range",
    "tide_range_mean",
]


# =========================
# Flow prep
# =========================


def build_daily_flow_table(
    flow_df: pd.DataFrame,
    valid_regions: Sequence[int],
) -> pd.DataFrame:
    out = flow_df.copy()
    out = filter_usgs_iv_qc(out)

    out = out.rename(columns={"site_id": "region", "date_time": "time", "value": "streamflow"})
    out = out[out["region"].isin(valid_regions)].copy()

    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True).dt.tz_localize(None)
    out["streamflow"] = pd.to_numeric(out["streamflow"], errors="coerce")
    out["no_data_value"] = pd.to_numeric(out["no_data_value"], errors="coerce")

    has_no_data = out["no_data_value"].notna()
    out = out.loc[~(has_no_data & (out["streamflow"] == out["no_data_value"]))].copy()

    out = out.dropna(subset=["region", "time", "streamflow"]).copy()
    out = out.sort_values(["region", "time"]).reset_index(drop=True)

    hourly = (
        out.set_index("time").groupby("region").resample("1h")["streamflow"].mean().reset_index()
    )

    hourly["date"] = hourly["time"].dt.floor("D")

    daily = hourly.groupby(["region", "date"], as_index=False).agg(
        streamflow_mean=("streamflow", "mean"),
        streamflow_median=("streamflow", "median"),
        streamflow_p90=("streamflow", lambda s: s.quantile(0.9)),
        streamflow_max=("streamflow", "max"),
        flow_n_obs=("streamflow", lambda s: s.notna().sum()),
    )

    daily = add_time_fields(daily, date_col="date")
    return daily


# =========================
# Tide prep
# =========================


def build_daily_tide_table(
    tide_df: pd.DataFrame,
    valid_regions: Sequence[int],
) -> pd.DataFrame:
    out = tide_df.copy()
    out = out.rename(columns={"site_id": "region", "tide_elevation": "tide"})
    out = out[out["region"].isin(valid_regions)].copy()

    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True).dt.tz_localize(None)
    out["tide"] = pd.to_numeric(out["tide"], errors="coerce")

    out = out.dropna(subset=["region", "time", "tide"]).copy()
    out["date"] = out["time"].dt.floor("D")

    daily = out.groupby(["region", "date"], as_index=False).agg(
        tide_mean=("tide", "mean"),
        tide_min=("tide", "min"),
        tide_max=("tide", "max"),
        tide_range=("tide", lambda s: s.max() - s.min()),
        tide_n_obs=("tide", lambda s: s.notna().sum()),
    )

    daily = add_time_fields(daily, date_col="date")
    return daily


# =========================
# Merge + aggregate
# =========================


def build_daily_driver_table(
    daily_p: pd.DataFrame,
    daily_wave: pd.DataFrame,
    daily_flow: pd.DataFrame,
    daily_tide: pd.DataFrame,
) -> pd.DataFrame:
    daily = (
        daily_p[
            [
                "region",
                "date",
                "p_open",
                "p_closed",
                "y_pred_open",
                "p_open_original",
                "p_closed_original",
                "y_pred_open_original",
                "entropy",
                "entropy_norm",
                "year",
                "month",
                "season",
                "water_year",
                "wy_season",
                "is_obs_original",
                "is_obs",
                "is_interpolated",
                "is_flicker_removed",
                "flicker_pattern",
                "flicker_previous_observation_date",
                "flicker_next_observation_date",
                "flicker_days_since_previous_observation",
                "flicker_days_until_next_observation",
                "days_since_previous_real_observation",
                "days_until_next_real_observation",
                "min_days_since_real_observation",
                "max_days_since_real_observation",
            ]
        ]
        .merge(
            daily_wave[
                [
                    "region",
                    "date",
                    "wave_hs_mean",
                    "wave_hs_p90",
                    "wave_hs_max",
                    "wave_tp_mean",
                    "wave_tp_p90",
                    "wave_tp_max",
                    "wave_energy_mean",
                    "wave_energy_p90",
                    "wave_energy_max",
                    "wave_energy_cross_mean",
                    "wave_energy_cross_p90",
                    "wave_energy_cross_max",
                    "wave_energy_along_mean",
                    "wave_energy_along_p90",
                    "wave_energy_along_max",
                    "wave_n_obs",
                ]
            ],
            on=["region", "date"],
            how="left",
        )
        .merge(
            daily_flow[
                [
                    "region",
                    "date",
                    "streamflow_mean",
                    "streamflow_median",
                    "streamflow_p90",
                    "streamflow_max",
                    "flow_n_obs",
                ]
            ],
            on=["region", "date"],
            how="left",
        )
        .merge(
            daily_tide[
                [
                    "region",
                    "date",
                    "tide_mean",
                    "tide_min",
                    "tide_max",
                    "tide_range",
                    "tide_n_obs",
                ]
            ],
            on=["region", "date"],
            how="left",
        )
    )

    daily = add_time_fields(daily, date_col="date")
    daily = add_per_region_rank_normalization(daily, DRIVER_RANK_COLUMNS)
    return daily


def aggregate_to_month(
    daily: pd.DataFrame,
    min_cov_p_open: float,
    min_cov_driver: float,
) -> pd.DataFrame:
    out = daily.groupby(["region", "year", "month"], as_index=False).agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_days=("date", "size"),
        original_non_interpolated_days=("is_obs_original", "sum"),
        non_interpolated_days=("is_obs", "sum"),
        flicker_removed_observations=("is_flicker_removed", "sum"),
        mean_p_open=("p_open", "mean"),
        mean_p_closed=("p_closed", "mean"),
        mean_p_open_original=("p_open_original", "mean"),
        mean_p_closed_original=("p_closed_original", "mean"),
        mean_entropy=("entropy_norm", "mean"),
        wave_hs_mean=("wave_hs_mean", "mean"),
        wave_hs_p90=("wave_hs_p90", "mean"),
        wave_tp_mean=("wave_tp_mean", "mean"),
        wave_tp_p90=("wave_tp_p90", "mean"),
        wave_energy_mean=("wave_energy_mean", "mean"),
        wave_energy_p90=("wave_energy_p90", "mean"),
        wave_energy_max=("wave_energy_max", "max"),
        wave_energy_cross_mean=("wave_energy_cross_mean", "mean"),
        wave_energy_cross_p90=("wave_energy_cross_p90", "mean"),
        wave_energy_cross_max=("wave_energy_cross_max", "max"),
        wave_energy_along_mean=("wave_energy_along_mean", "mean"),
        wave_energy_along_p90=("wave_energy_along_p90", "mean"),
        wave_energy_along_max=("wave_energy_along_max", "max"),
        streamflow_mean=("streamflow_mean", "mean"),
        streamflow_median=("streamflow_median", "mean"),
        streamflow_p90=("streamflow_p90", "mean"),
        streamflow_max=("streamflow_max", "max"),
        tide_mean=("tide_mean", "mean"),
        tide_range_mean=("tide_range", "mean"),
        tide_range_max=("tide_range", "max"),
        p_open_cov=("p_open", lambda s: s.notna().mean()),
        wave_cov=("wave_energy_mean", lambda s: s.notna().mean()),
        flow_cov=("streamflow_mean", lambda s: s.notna().mean()),
        tide_cov=("tide_range", lambda s: s.notna().mean()),
    )

    out["water_year"] = np.where(out["month"] >= 10, out["year"] + 1, out["year"])

    month_to_season = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall",
        10: "fall",
        11: "fall",
    }
    out["season"] = out["month"].map(month_to_season)

    def month_to_wy_season(month: int) -> str:
        if month in [10, 11, 12]:
            return "OND"
        if month in [1, 2, 3]:
            return "JFM"
        if month in [4, 5, 6]:
            return "AMJ"
        return "JAS"

    out["wy_season"] = out["month"].map(month_to_wy_season)

    out["keep"] = (
        (out["p_open_cov"] >= min_cov_p_open)
        & (out["wave_cov"] >= min_cov_driver)
        & (out["flow_cov"] >= min_cov_driver)
        & (out["tide_cov"] >= min_cov_driver)
    )

    out = add_per_region_rank_normalization(out, DRIVER_RANK_COLUMNS)

    return out


def aggregate_to_period(
    daily: pd.DataFrame,
    period_col: str,
    min_cov_p_open: float,
    min_cov_driver: float,
) -> pd.DataFrame:
    if period_col == "season":
        group_cols = ["region", "year", "season"]
    elif period_col == "wy_season":
        group_cols = ["region", "water_year", "wy_season"]
    elif period_col == "water_year":
        group_cols = ["region", "water_year"]
    else:
        raise ValueError("period_col must be 'season', 'wy_season', or 'water_year'")

    out = daily.groupby(group_cols, as_index=False).agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_days=("date", "size"),
        original_non_interpolated_days=("is_obs_original", "sum"),
        non_interpolated_days=("is_obs", "sum"),
        flicker_removed_observations=("is_flicker_removed", "sum"),
        mean_p_open=("p_open", "mean"),
        mean_p_closed=("p_closed", "mean"),
        mean_p_open_original=("p_open_original", "mean"),
        mean_p_closed_original=("p_closed_original", "mean"),
        mean_entropy=("entropy_norm", "mean"),
        wave_hs_mean=("wave_hs_mean", "mean"),
        wave_hs_p90=("wave_hs_p90", "mean"),
        wave_tp_mean=("wave_tp_mean", "mean"),
        wave_tp_p90=("wave_tp_p90", "mean"),
        wave_energy_mean=("wave_energy_mean", "mean"),
        wave_energy_p90=("wave_energy_p90", "mean"),
        wave_energy_max=("wave_energy_max", "max"),
        wave_energy_cross_mean=("wave_energy_cross_mean", "mean"),
        wave_energy_cross_p90=("wave_energy_cross_p90", "mean"),
        wave_energy_cross_max=("wave_energy_cross_max", "max"),
        wave_energy_along_mean=("wave_energy_along_mean", "mean"),
        wave_energy_along_p90=("wave_energy_along_p90", "mean"),
        wave_energy_along_max=("wave_energy_along_max", "max"),
        streamflow_mean=("streamflow_mean", "mean"),
        streamflow_median=("streamflow_median", "mean"),
        streamflow_p90=("streamflow_p90", "mean"),
        streamflow_max=("streamflow_max", "max"),
        tide_mean=("tide_mean", "mean"),
        tide_range_mean=("tide_range", "mean"),
        tide_range_max=("tide_range", "max"),
        p_open_cov=("p_open", lambda s: s.notna().mean()),
        wave_cov=("wave_energy_mean", lambda s: s.notna().mean()),
        flow_cov=("streamflow_mean", lambda s: s.notna().mean()),
        tide_cov=("tide_range", lambda s: s.notna().mean()),
    )

    out["keep"] = (
        (out["p_open_cov"] >= min_cov_p_open)
        & (out["wave_cov"] >= min_cov_driver)
        & (out["flow_cov"] >= min_cov_driver)
        & (out["tide_cov"] >= min_cov_driver)
    )

    out = add_per_region_rank_normalization(out, DRIVER_RANK_COLUMNS)

    return out


# =========================
# Main
# =========================


def main() -> None:
    cfg = SeasonalDriverConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    preds_all = read_table(cfg.predictions_path)
    print(f"Predictions rows: {len(preds_all):,}")

    print("Building daily p_open table...")
    daily_p = build_daily_p_open_table(
        preds_all=preds_all,
        valid_regions=cfg.valid_regions,
        open_threshold=cfg.open_threshold,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        flicker_window_days=cfg.flicker_window_days,
    )
    print(f"daily_p rows: {len(daily_p):,}")

    print("Loading wave lookup tables...")
    nearest_cdip_df = pd.read_csv(cfg.nearest_cdip_path)
    cdip_points_meta_df = pd.read_csv(cfg.cdip_points_meta_path)
    wave_orientation_lookup = prepare_wave_orientation_lookup(
        nearest_cdip_df=nearest_cdip_df,
        cdip_points_meta_df=cdip_points_meta_df,
    )

    print("Loading waves...")
    waves_raw = read_table(cfg.waves_path)
    print(f"Raw wave rows: {len(waves_raw):,}")

    print("Preparing waves...")
    waves_prepped = prepare_waves_for_daily(
        waves_df=waves_raw,
        wave_orientation_lookup=wave_orientation_lookup,
        valid_regions=cfg.valid_regions,
    )
    print(f"Prepared wave rows: {len(waves_prepped):,}")

    print("Building daily wave table...")
    daily_wave = build_daily_wave_table(
        waves=waves_prepped,
        valid_regions=cfg.valid_regions,
    )
    print(f"daily_wave rows: {len(daily_wave):,}")

    print("Loading streamflow...")
    flow_raw = read_table(cfg.streamflow_path)
    print(f"Raw flow rows: {len(flow_raw):,}")

    print("Building daily flow table...")
    daily_flow = build_daily_flow_table(
        flow_df=flow_raw,
        valid_regions=cfg.valid_regions,
    )
    print(f"daily_flow rows: {len(daily_flow):,}")

    print("Loading tide...")
    tide_raw = read_table(cfg.tide_path)
    print(f"Raw tide rows: {len(tide_raw):,}")

    print("Building daily tide table...")
    daily_tide = build_daily_tide_table(
        tide_df=tide_raw,
        valid_regions=cfg.valid_regions,
    )
    print(f"daily_tide rows: {len(daily_tide):,}")

    print("Merging daily driver table...")
    daily_driver = build_daily_driver_table(
        daily_p=daily_p,
        daily_wave=daily_wave,
        daily_flow=daily_flow,
        daily_tide=daily_tide,
    )
    print(f"daily_driver rows: {len(daily_driver):,}")

    print("Aggregating seasonal table...")
    seasonal = aggregate_to_period(
        daily=daily_driver,
        period_col="season",
        min_cov_p_open=cfg.min_coverage_p_open,
        min_cov_driver=cfg.min_coverage_driver,
    )

    print("Aggregating water-year seasonal table...")
    wy_seasonal = aggregate_to_period(
        daily=daily_driver,
        period_col="wy_season",
        min_cov_p_open=cfg.min_coverage_p_open,
        min_cov_driver=cfg.min_coverage_driver,
    )

    print("Aggregating annual table...")
    annual = aggregate_to_period(
        daily=daily_driver,
        period_col="water_year",
        min_cov_p_open=cfg.min_coverage_p_open,
        min_cov_driver=cfg.min_coverage_driver,
    )

    print("Aggregating monthly table...")
    monthly = aggregate_to_month(
        daily=daily_driver,
        min_cov_p_open=cfg.min_coverage_p_open,
        min_cov_driver=cfg.min_coverage_driver,
    )

    print("Saving outputs...")
    daily_driver.to_parquet(out_dir / "daily_driver_table.parquet", index=False)
    seasonal.to_parquet(out_dir / "seasonal_driver_table.parquet", index=False)
    wy_seasonal.to_parquet(out_dir / "wy_seasonal_driver_table.parquet", index=False)
    annual.to_parquet(out_dir / "annual_driver_table.parquet", index=False)
    monthly.to_parquet(out_dir / "monthly_driver_table.parquet", index=False)

    print("Done.")
    print(f"Saved daily_driver_table.parquet to {out_dir}")
    print(f"Saved seasonal_driver_table.parquet to {out_dir}")
    print(f"Saved wy_seasonal_driver_table.parquet to {out_dir}")
    print(f"Saved annual_driver_table.parquet to {out_dir}")
    print(f"Saved monthly_driver_table.parquet to {out_dir}")


if __name__ == "__main__":
    main()
