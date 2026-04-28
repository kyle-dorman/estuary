from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# =========================
# Configuration
# =========================


@dataclass
class FeatureConfig:
    show_progress: bool = True
    save_outputs: bool = True

    # Window lengths in days.
    backward_days: tuple[int, ...] = (7,)
    forward_days: tuple[int, ...] = (3, 7, 14)

    # Expected observations per day for coverage metrics.
    expected_obs_per_day_wave: int = 24
    expected_obs_per_day_flow: int = 24 * 4
    expected_obs_per_day_tide: int = 24 * 2
    expected_obs_per_day_physics: int = 24


# =========================
# Progress helpers
# =========================


def iter_items(
    items: Iterable,
    total: int,
    desc: str,
    enabled: bool,
):
    """Wrap iteration in tqdm if available."""
    if enabled and tqdm is not None:
        return tqdm(items, total=total, desc=desc)
    return items


def print_progress(enabled: bool, message: str) -> None:
    """Fallback progress messages if tqdm is unavailable."""
    if enabled and tqdm is None:
        print(message)


# =========================
# Path helpers
# =========================


def get_output_path(anchors_path: str) -> Path:
    """
    Save output in the same folder as the anchors table.
    """
    anchors_file = Path(anchors_path)
    stem = anchors_file.stem
    output_dir = anchors_file.parent

    feature_table_path = output_dir / f"{stem}_feature_table.parquet"

    return feature_table_path


# =========================
# Validation helpers
# =========================


def validate_anchor_columns(df: pd.DataFrame) -> None:
    required = {"region", "anchor_time", "horizon_days", "event_y", "censored"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Anchors missing required columns: {sorted(missing)}")


def validate_wave_columns(df: pd.DataFrame) -> None:
    required = {
        "site_id",
        "time",
        "wave_hs",
        "wave_tp",
        "wave_dp",
        "wave_flag_primary",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Waves missing required columns: {sorted(missing)}")


def validate_cdip_points_meta_columns(df: pd.DataFrame) -> None:
    required = {"cdip_id", "shore_normal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CDIP points meta missing required columns: {sorted(missing)}")


def validate_nearest_cdip_columns(df: pd.DataFrame) -> None:
    required = {"site_id", "cdip_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Nearest CDIP points missing required columns: {sorted(missing)}")


def validate_tide_columns(df: pd.DataFrame) -> None:
    required = {"site_id", "time", "tide_elevation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tide missing required columns: {sorted(missing)}")


def validate_flow_columns(df: pd.DataFrame) -> None:
    required = {
        "site_id",
        "date_time",
        "variable_code",
        "value",
        "no_data_value",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Streamflow missing required columns: {sorted(missing)}")


# =========================
# CDIP wave QC filter
# =========================


def filter_cdip_wave_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a conservative CDIP QC filter.

    CDIP primary QC flags follow the UNESCO/IOC convention:
    1 = good
    2 = not_evaluated
    3 = questionable
    4 = bad
    9 = missing

    For a first pass, keep 1/2/3 and drop 4/9.
    We do not filter on the secondary flag here because its meanings can vary
    by product, while the primary flag consistently identifies bad/missing data.
    """
    out = df.copy()
    out["wave_flag_primary"] = pd.to_numeric(out["wave_flag_primary"], errors="coerce")
    out = out[out["wave_flag_primary"].isin([1, 2, 3])].copy()
    return out


def filter_usgs_iv_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a conservative USGS instantaneous-values filter.

    Rules for v1:
    - keep discharge only: variable_code == "00060"
    - if an approval_status column is present, keep only Approved rows
    - otherwise, fall back to qualifier_codes and keep rows that appear approved
      (contain "A") while dropping rows that appear provisional (contain "P")

    This fallback exists because some exported IV tables expose approval through
    qualifier codes rather than a separate approval_status column.
    """
    out = df.copy()

    out["variable_code"] = out["variable_code"].astype(str)
    out = out[out["variable_code"] == "00060"].copy()

    if "approval_status" in out.columns:
        approval = out["approval_status"].astype(str).str.strip().str.lower()
        out = out[approval == "approved"].copy()
    else:
        qualifier = out.get("qualifier_codes", pd.Series("", index=out.index))
        qualifier = qualifier.fillna("").astype(str).str.upper()
        has_a = qualifier.str.contains(r"(?:^|\\b|,|;)A(?:\\b|,|;|$)", regex=True)
        has_p = qualifier.str.contains(r"(?:^|\\b|,|;)P(?:\\b|,|;|$)", regex=True)
        out = out[has_a & ~has_p].copy()

    return out


def prepare_wave_orientation_lookup(
    nearest_cdip_df: pd.DataFrame,
    cdip_points_meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a region-to-CDIP lookup with shoreline normal.

    nearest_cdip_df provides the estuary site_id <-> cdip_id mapping.
    cdip_points_meta_df provides shore_normal by cdip_id.
    """
    validate_nearest_cdip_columns(nearest_cdip_df)
    validate_cdip_points_meta_columns(cdip_points_meta_df)

    nearest = nearest_cdip_df[["site_id", "cdip_id"]].copy().drop_duplicates()
    meta = cdip_points_meta_df[["cdip_id", "shore_normal"]].copy().drop_duplicates()
    meta["shore_normal"] = pd.to_numeric(meta["shore_normal"], errors="coerce")

    out = nearest.merge(meta, on="cdip_id", how="left")
    out = out.rename(columns={"site_id": "region"})
    out = out.dropna(subset=["region", "cdip_id", "shore_normal"]).copy()
    out = out.drop_duplicates(subset=["region", "cdip_id"])

    return out[["region", "shore_normal"]]


def wrap_angle_degrees(series: pd.Series) -> pd.Series:
    """
    Wrap angles to [-180, 180) degrees.
    """
    s = pd.to_numeric(series, errors="coerce")
    return ((s + 180.0) % 360.0) - 180.0


# =========================
# Standardization
# =========================


# Helper for anchor time localizing to 11am, then UTC, then naive
def set_anchor_time_local_11am(series: pd.Series) -> pd.Series:
    """
    Convert a date-like series to ~11:00 AM America/Los_Angeles,
    then convert to UTC and drop timezone.
    """
    s = pd.to_datetime(series, errors="coerce")
    s = s.dt.tz_localize(None)
    s = s.dt.floor("D") + pd.Timedelta(hours=11)
    s = s.dt.tz_localize("America/Los_Angeles")
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def prepare_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize anchors table.
    """
    validate_anchor_columns(df)

    out = df.copy()
    out["anchor_time"] = set_anchor_time_local_11am(out["anchor_time"])
    if out["anchor_time"].isna().any():
        raise ValueError("Found invalid anchor_time values.")

    out = out.sort_values(["region", "anchor_time", "horizon_days"]).reset_index(drop=True)
    return out


def prepare_waves(
    df: pd.DataFrame,
    wave_orientation_lookup: pd.DataFrame,
    valid_regions: list[int],
) -> pd.DataFrame:
    """
    Standardize waves table, apply a conservative CDIP QC filter, and derive
    total, cross-shore, and alongshore wave-forcing proxies.

    Wave-forcing proxy uses Hs^2 * Tp. Directional components use the angle
    relative to shoreline normal.
    """
    validate_wave_columns(df)

    out = df.copy()
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

    # Deep-water wave-forcing proxy, ignoring constants common to all rows:
    # forcing ~ Hs^2 * Tp
    out["wave_forcing"] = (out["wave_hs"] ** 2) * out["wave_tp"]

    # Relative angle to shoreline normal. Because wave_dp is a direction field
    # and shoreline-normal conventions can vary, start with absolute directional
    # magnitudes for cross-shore and alongshore components.
    rel_deg = wrap_angle_degrees(out["wave_dp"] - out["shore_normal"])
    rel_rad = np.deg2rad(rel_deg)

    out["wave_forcing_cross"] = out["wave_forcing"] * np.abs(np.cos(rel_rad))
    out["wave_forcing_along"] = out["wave_forcing"] * np.abs(np.sin(rel_rad))

    out = out.sort_values(["region", "time"]).reset_index(drop=True)

    return out[
        [
            "region",
            "time",
            "wave_hs",
            "wave_tp",
            "wave_dp",
            "shore_normal",
            "wave_forcing",
            "wave_forcing_cross",
            "wave_forcing_along",
            "wave_flag_primary",
        ]
    ]


def prepare_tide(df: pd.DataFrame, valid_regions: list[int]) -> pd.DataFrame:
    """
    Standardize tide table.
    """
    validate_tide_columns(df)

    out = df.copy()
    out = out.rename(columns={"site_id": "region", "tide_elevation": "tide"})
    out = out[out["region"].isin(valid_regions)].copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True).dt.tz_localize(None)
    out["tide"] = pd.to_numeric(out["tide"], errors="coerce")

    out = out.dropna(subset=["region", "time", "tide"]).copy()
    out = out.sort_values(["region", "time"]).reset_index(drop=True)

    return out[["region", "time", "tide"]]


def prepare_streamflow(df: pd.DataFrame, valid_regions: list[int]) -> pd.DataFrame:
    """
    Standardize USGS instantaneous streamflow and apply conservative IV filtering,
    including removal of no-data sentinel values.
    """
    validate_flow_columns(df)

    out = df.copy()
    out = filter_usgs_iv_qc(out)
    out = out.rename(columns={"site_id": "region", "date_time": "time", "value": "streamflow"})
    out = out[out["region"].isin(valid_regions)].copy()

    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True).dt.tz_localize(None)
    out["streamflow"] = pd.to_numeric(out["streamflow"], errors="coerce")

    out["no_data_value"] = pd.to_numeric(out["no_data_value"], errors="coerce")

    # Drop rows where the streamflow equals the USGS no-data sentinel value.
    has_no_data = out["no_data_value"].notna()
    out = out.loc[~(has_no_data & (out["streamflow"] == out["no_data_value"]))].copy()

    out = out.dropna(subset=["region", "time", "streamflow"]).copy()
    out["flow_forcing"] = transform_streamflow_to_forcing(out["streamflow"])
    out = out.sort_values(["region", "time"]).reset_index(drop=True)

    return out[["region", "time", "streamflow", "flow_forcing", "variable_code"]]


# =========================
# Basic utilities
# =========================


def get_region_slice(df: pd.DataFrame, region: int | str) -> pd.DataFrame:
    """
    Return one region sorted by time.
    """
    sub = df.loc[df["region"] == region].copy()
    sub = sub.sort_values("time").reset_index(drop=True)
    return sub


def fraction_coverage(n_obs: int, expected_obs: int) -> float:
    """
    Coverage fraction in [0, 1+] depending on duplicates / over-sampling.
    """
    if expected_obs <= 0:
        return np.nan
    return n_obs / expected_obs


def get_time_window_mask(
    times: pd.Series,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    include_start: bool,
    include_end: bool,
) -> pd.Series:
    """
    Build a boolean mask for a time window.
    """
    if include_start:
        left = times >= start_time
    else:
        left = times > start_time

    if include_end:
        right = times <= end_time
    else:
        right = times < end_time

    return left & right


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else np.nan


def safe_max(series: pd.Series) -> float:
    return float(series.max()) if len(series) > 0 else np.nan


def safe_min(series: pd.Series) -> float:
    return float(series.min()) if len(series) > 0 else np.nan


def build_hourly_wave_flow_physics_table(
    waves: pd.DataFrame,
    flow: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align wave and flow forcing to a common hourly grid within each region,
    then compute instantaneous log wave/flow physics indexes.

    We use hourly means for alignment, then summarize these instantaneous
    physics-index time series within anchor windows.
    """
    print("Building hourly wave-flow physics table...")

    wave_hourly_frames: list[pd.DataFrame] = []
    flow_hourly_frames: list[pd.DataFrame] = []

    wave_groups = list(waves.groupby("region", sort=False))
    print(f"Resampling waves to hourly: {len(wave_groups):,} regions")

    for region, sub in wave_groups:
        sub = sub.sort_values("time").copy()
        sub_hourly = (
            sub.set_index("time")
            .resample("1h")
            .agg(
                wave_forcing=("wave_forcing", "mean"),
                wave_forcing_cross=("wave_forcing_cross", "mean"),
                wave_forcing_along=("wave_forcing_along", "mean"),
            )
            .reset_index()
        )
        sub_hourly["region"] = region
        wave_hourly_frames.append(sub_hourly)

    flow_groups = list(flow.groupby("region", sort=False))
    print(f"Resampling streamflow to hourly: {len(flow_groups):,} regions")

    for region, sub in flow_groups:
        sub = sub.sort_values("time").copy()
        sub_hourly = (
            sub.set_index("time")
            .resample("1h")
            .agg(flow_forcing=("flow_forcing", "mean"))
            .reset_index()
        )
        sub_hourly["region"] = region
        flow_hourly_frames.append(sub_hourly)

    if wave_hourly_frames:
        wave_hourly = pd.concat(wave_hourly_frames, ignore_index=True)
    else:
        wave_hourly = pd.DataFrame(
            columns=["time", "wave_forcing", "wave_forcing_cross", "wave_forcing_along", "region"]
        )

    if flow_hourly_frames:
        flow_hourly = pd.concat(flow_hourly_frames, ignore_index=True)
    else:
        flow_hourly = pd.DataFrame(columns=["time", "flow_forcing", "region"])

    print(f"Hourly wave rows: {len(wave_hourly):,}")
    print(f"Hourly flow rows: {len(flow_hourly):,}")
    print("Merging hourly wave and flow tables...")

    physics = wave_hourly.merge(flow_hourly, on=["region", "time"], how="inner")
    print(f"Merged hourly physics rows before dropna: {len(physics):,}")

    physics = physics.dropna(
        subset=["wave_forcing", "wave_forcing_cross", "wave_forcing_along", "flow_forcing"]
    ).copy()

    print(f"Merged hourly physics rows after dropna: {len(physics):,}")
    print("Computing instantaneous physics indexes...")

    physics["physics_index_inst"] = safe_log(physics["wave_forcing"]) - safe_log(
        physics["flow_forcing"],
        eps=1e-2,
    )
    physics["physics_index_cross_inst"] = safe_log(physics["wave_forcing_cross"]) - safe_log(
        physics["flow_forcing"],
        eps=1e-2,
    )
    physics["physics_index_along_inst"] = safe_log(physics["wave_forcing_along"]) - safe_log(
        physics["flow_forcing"],
        eps=1e-2,
    )

    physics = physics.sort_values(["region", "time"]).reset_index(drop=True)
    print(f"Finished hourly wave-flow physics table: {len(physics):,} rows")

    return physics


def summarize_physics_index_window(
    physics_region: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    expected_obs_per_day: int,
    explicit_suffix: str,
    include_start: bool,
    include_end: bool,
) -> dict:
    """
    Summarize instantaneous hourly physics-index time series within a window.
    """
    out = {}

    mask = get_time_window_mask(
        physics_region["time"],
        start_time=start_time,
        end_time=end_time,
        include_start=include_start,
        include_end=include_end,
    )
    sub = physics_region.loc[mask]

    expected_obs = int(((end_time - start_time).total_seconds() / 86400.0) * expected_obs_per_day)

    out[f"physics_index_inst_mean_{explicit_suffix}"] = safe_mean(sub["physics_index_inst"])
    out[f"physics_index_inst_min_{explicit_suffix}"] = safe_min(sub["physics_index_inst"])
    out[f"physics_index_inst_max_{explicit_suffix}"] = safe_max(sub["physics_index_inst"])

    out[f"physics_index_cross_inst_mean_{explicit_suffix}"] = safe_mean(
        sub["physics_index_cross_inst"]
    )
    out[f"physics_index_cross_inst_min_{explicit_suffix}"] = safe_min(
        sub["physics_index_cross_inst"]
    )
    out[f"physics_index_cross_inst_max_{explicit_suffix}"] = safe_max(
        sub["physics_index_cross_inst"]
    )

    out[f"physics_index_along_inst_mean_{explicit_suffix}"] = safe_mean(
        sub["physics_index_along_inst"]
    )
    out[f"physics_index_along_inst_min_{explicit_suffix}"] = safe_min(
        sub["physics_index_along_inst"]
    )
    out[f"physics_index_along_inst_max_{explicit_suffix}"] = safe_max(
        sub["physics_index_along_inst"]
    )

    out[f"physics_index_coverage_{explicit_suffix}"] = fraction_coverage(len(sub), expected_obs)

    return out


def transform_streamflow_to_forcing(series: pd.Series) -> pd.Series:
    """
    Convert discharge to a simple superlinear river-forcing proxy.

    For v1, use Q^1.5 as a pragmatic approximation to increasing hydraulic
    work / flushing capacity with discharge.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.clip(lower=0) ** 1.5


# =========================
# Additional helper functions
# =========================


def safe_log(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    """
    Elementwise natural log with a small positive floor.
    """
    s = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.log(s.clip(lower=eps)))


def add_log_feature_if_present(df: pd.DataFrame, col: str, eps: float = 1e-6) -> None:
    """
    Add a log-transformed version of a feature in-place if the source column exists.
    """
    if col in df.columns:
        df[f"log_{col}"] = safe_log(df[col], eps=eps)


def add_physics_index_if_present(
    df: pd.DataFrame,
    wave_col: str,
    flow_col: str,
    out_col: str,
    eps: float = 1e-6,
) -> None:
    """
    Add a simple physics index in-place if both source columns exist.

    physics_index = log(wave_metric) - log(flow_metric)
    """
    if wave_col in df.columns and flow_col in df.columns:
        df[out_col] = safe_log(df[wave_col], eps=eps) - safe_log(
            df[flow_col],
            eps=1e-2,
        )


# =========================
# Window feature builders
# =========================


def summarize_wave_window(
    wave_region: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    expected_obs_per_day: int,
    explicit_suffix: str,
    include_start: bool,
    include_end: bool,
) -> dict:
    """
    Compute wave forcing features for a given window.
    """
    out = {}

    mask = get_time_window_mask(
        wave_region["time"],
        start_time=start_time,
        end_time=end_time,
        include_start=include_start,
        include_end=include_end,
    )
    sub = wave_region.loc[mask]

    expected_obs = int(((end_time - start_time).total_seconds() / 86400.0) * expected_obs_per_day)

    out[f"wave_forcing_mean_{explicit_suffix}"] = safe_mean(sub["wave_forcing"])
    out[f"wave_forcing_min_{explicit_suffix}"] = safe_min(sub["wave_forcing"])
    out[f"wave_forcing_max_{explicit_suffix}"] = safe_max(sub["wave_forcing"])

    out[f"wave_forcing_cross_mean_{explicit_suffix}"] = safe_mean(sub["wave_forcing_cross"])
    out[f"wave_forcing_cross_min_{explicit_suffix}"] = safe_min(sub["wave_forcing_cross"])
    out[f"wave_forcing_cross_max_{explicit_suffix}"] = safe_max(sub["wave_forcing_cross"])

    out[f"wave_forcing_along_mean_{explicit_suffix}"] = safe_mean(sub["wave_forcing_along"])
    out[f"wave_forcing_along_min_{explicit_suffix}"] = safe_min(sub["wave_forcing_along"])
    out[f"wave_forcing_along_max_{explicit_suffix}"] = safe_max(sub["wave_forcing_along"])

    out[f"wave_coverage_{explicit_suffix}"] = fraction_coverage(len(sub), expected_obs)

    return out


def summarize_streamflow_window(
    flow_region: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    expected_obs_per_day: int,
    explicit_suffix: str,
    include_start: bool,
    include_end: bool,
) -> dict:
    """
    Compute transformed flow-forcing features for a given window.
    """
    out = {}

    mask = get_time_window_mask(
        flow_region["time"],
        start_time=start_time,
        end_time=end_time,
        include_start=include_start,
        include_end=include_end,
    )
    sub = flow_region.loc[mask]

    expected_obs = max(
        1,
        int(np.ceil(((end_time - start_time).total_seconds() / 86400.0) * expected_obs_per_day)),
    )

    out[f"flow_forcing_mean_{explicit_suffix}"] = safe_mean(sub["flow_forcing"])
    out[f"flow_forcing_min_{explicit_suffix}"] = safe_min(sub["flow_forcing"])
    out[f"flow_forcing_max_{explicit_suffix}"] = safe_max(sub["flow_forcing"])
    out[f"flow_coverage_{explicit_suffix}"] = fraction_coverage(len(sub), expected_obs)

    return out


def summarize_tide_window(
    tide_region: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    expected_obs_per_day: int,
    explicit_suffix: str,
    include_start: bool,
    include_end: bool,
) -> dict:
    """
    Compute tide features for a given window.
    """
    out = {}

    mask = get_time_window_mask(
        tide_region["time"],
        start_time=start_time,
        end_time=end_time,
        include_start=include_start,
        include_end=include_end,
    )
    sub = tide_region.loc[mask]

    expected_obs = int(((end_time - start_time).total_seconds() / 86400.0) * expected_obs_per_day)

    tide_mean = safe_mean(sub["tide"])
    tide_min = safe_min(sub["tide"])
    tide_max = safe_max(sub["tide"])

    tide_range = np.nan
    if len(sub) > 0:
        tide_range = float(sub["tide"].max() - sub["tide"].min())

    out[f"tide_mean_{explicit_suffix}"] = tide_mean
    out[f"tide_min_{explicit_suffix}"] = tide_min
    out[f"tide_max_{explicit_suffix}"] = tide_max
    out[f"tide_range_{explicit_suffix}"] = tide_range
    out[f"tide_coverage_{explicit_suffix}"] = fraction_coverage(len(sub), expected_obs)

    return out


# =========================
# Anchor-state helpers
# =========================


def build_anchor_state_lookup(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Minimal lookup table for p_open at anchor time.

    Expects:
    - region
    - date
    - p_open
    """
    required = {"region", "date", "p_open"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Predictions dataframe missing columns: {sorted(missing)}")

    out = predictions_df.copy()
    out["date"] = set_anchor_time_local_11am(out["date"])
    out = out.dropna(subset=["region", "date", "p_open"]).copy()

    out = out.rename(columns={"date": "anchor_time", "p_open": "p_open_anchor"})
    out = out[["region", "anchor_time", "p_open_anchor"]].drop_duplicates()
    out = out.sort_values(["region", "anchor_time"]).reset_index(drop=True)

    return out


# =========================
# Per-anchor feature extraction
# =========================


def build_feature_row(
    anchor_row: pd.Series,
    wave_region: pd.DataFrame,
    tide_region: pd.DataFrame,
    flow_region: pd.DataFrame,
    physics_region: pd.DataFrame,
    config: FeatureConfig,
) -> dict:
    """
    Build all environmental features for a single anchor row.
    """
    anchor_time = pd.Timestamp(anchor_row["anchor_time"])

    row = {
        "region": anchor_row["region"],
        "anchor_time": anchor_time,
        "horizon_days": anchor_row["horizon_days"],
        "event_y": anchor_row["event_y"],
        "censored": anchor_row["censored"],
    }

    # Carry through optional columns if present.
    for col in [
        "time_to_closure_days",
        "linked_event_time",
        "linked_event_midpoint_time",
        "linked_closure_score",
        "linked_motif_label",
    ]:
        if col in anchor_row.index:
            row[col] = anchor_row[col]

    # Backward windows.
    for days in config.backward_days:
        start_time = anchor_time - pd.Timedelta(days=days)
        end_time = anchor_time
        suffix = f"prev_{days}d"

        row.update(
            summarize_wave_window(
                wave_region=wave_region,
                start_time=start_time,
                end_time=end_time,
                expected_obs_per_day=config.expected_obs_per_day_wave,
                explicit_suffix=suffix,
                include_start=False,
                include_end=True,
            )
        )
        row.update(
            summarize_streamflow_window(
                flow_region=flow_region,
                start_time=start_time,
                end_time=end_time,
                expected_obs_per_day=config.expected_obs_per_day_flow,
                explicit_suffix=suffix,
                include_start=False,
                include_end=True,
            )
        )
        row.update(
            summarize_tide_window(
                tide_region=tide_region,
                start_time=start_time,
                end_time=end_time,
                expected_obs_per_day=config.expected_obs_per_day_tide,
                explicit_suffix=suffix,
                include_start=False,
                include_end=True,
            )
        )
        row.update(
            summarize_physics_index_window(
                physics_region=physics_region,
                start_time=start_time,
                end_time=end_time,
                expected_obs_per_day=config.expected_obs_per_day_physics,
                explicit_suffix=suffix,
                include_start=False,
                include_end=True,
            )
        )

    # Forward windows.
    for days in config.forward_days:
        start_time = anchor_time
        end_time = anchor_time + pd.Timedelta(days=days)
        suffix = f"fwd_{days}d"

        row.update(
            summarize_wave_window(
                wave_region=wave_region,
                start_time=start_time,
                end_time=end_time,
                expected_obs_per_day=config.expected_obs_per_day_wave,
                explicit_suffix=suffix,
                include_start=False,
                include_end=True,
            )
        )
        row.update(
            summarize_streamflow_window(
                flow_region=flow_region,
                start_time=start_time,
                end_time=end_time,
                expected_obs_per_day=config.expected_obs_per_day_flow,
                explicit_suffix=suffix,
                include_start=False,
                include_end=True,
            )
        )
        row.update(
            summarize_tide_window(
                tide_region=tide_region,
                start_time=start_time,
                end_time=end_time,
                expected_obs_per_day=config.expected_obs_per_day_tide,
                explicit_suffix=suffix,
                include_start=False,
                include_end=True,
            )
        )
        row.update(
            summarize_physics_index_window(
                physics_region=physics_region,
                start_time=start_time,
                end_time=end_time,
                expected_obs_per_day=config.expected_obs_per_day_physics,
                explicit_suffix=suffix,
                include_start=False,
                include_end=True,
            )
        )

    return row


def build_features_for_region(
    anchors_region: pd.DataFrame,
    wave_region: pd.DataFrame,
    tide_region: pd.DataFrame,
    flow_region: pd.DataFrame,
    physics_region: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Build feature rows for one region.
    """
    rows: list[dict] = []

    anchor_iter = iter_items(
        anchors_region.iterrows(),
        total=len(anchors_region),
        desc=f"Anchors region {anchors_region['region'].iloc[0]}",
        enabled=False,  # region-level progress is usually enough
    )

    for _, anchor_row in anchor_iter:
        rows.append(
            build_feature_row(
                anchor_row=anchor_row,
                wave_region=wave_region,
                tide_region=tide_region,
                flow_region=flow_region,
                physics_region=physics_region,
                config=config,
            )
        )

    return pd.DataFrame(rows)


# =========================
# Derived feature columns (logs, physics index)
# =========================


def add_derived_feature_columns(feature_table: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    Add log-transformed forcing features and simple physics-index combinations.
    """
    out = feature_table.copy()

    for days in config.backward_days:
        suffix = f"prev_{days}d"
        for col in [
            f"wave_forcing_mean_{suffix}",
            f"wave_forcing_min_{suffix}",
            f"wave_forcing_max_{suffix}",
            f"wave_forcing_cross_mean_{suffix}",
            f"wave_forcing_cross_min_{suffix}",
            f"wave_forcing_cross_max_{suffix}",
            f"wave_forcing_along_mean_{suffix}",
            f"wave_forcing_along_min_{suffix}",
            f"wave_forcing_along_max_{suffix}",
            f"flow_forcing_mean_{suffix}",
            f"flow_forcing_min_{suffix}",
            f"flow_forcing_max_{suffix}",
        ]:
            if "flow" in col:
                eps = 1e-2
            else:
                eps = 1e-6
            add_log_feature_if_present(out, col, eps=eps)

    for days in config.forward_days:
        suffix = f"fwd_{days}d"
        for col in [
            f"wave_forcing_mean_{suffix}",
            f"wave_forcing_min_{suffix}",
            f"wave_forcing_max_{suffix}",
            f"wave_forcing_cross_mean_{suffix}",
            f"wave_forcing_cross_min_{suffix}",
            f"wave_forcing_cross_max_{suffix}",
            f"wave_forcing_along_mean_{suffix}",
            f"wave_forcing_along_min_{suffix}",
            f"wave_forcing_along_max_{suffix}",
            f"flow_forcing_mean_{suffix}",
            f"flow_forcing_min_{suffix}",
            f"flow_forcing_max_{suffix}",
        ]:
            if "flow" in col:
                eps = 1e-2
            else:
                eps = 1e-6
            add_log_feature_if_present(out, col, eps=eps)

        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_mean_{suffix}",
            flow_col=f"flow_forcing_mean_{suffix}",
            out_col=f"physics_index_wave_forcing_mean_over_flow_forcing_mean_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_mean_{suffix}",
            flow_col=f"flow_forcing_min_{suffix}",
            out_col=f"physics_index_wave_forcing_mean_over_flow_forcing_min_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_max_{suffix}",
            flow_col=f"flow_forcing_mean_{suffix}",
            out_col=f"physics_index_wave_forcing_max_over_flow_forcing_mean_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_max_{suffix}",
            flow_col=f"flow_forcing_min_{suffix}",
            out_col=f"physics_index_wave_forcing_max_over_flow_forcing_min_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_cross_mean_{suffix}",
            flow_col=f"flow_forcing_mean_{suffix}",
            out_col=f"physics_index_wave_forcing_cross_mean_over_flow_forcing_mean_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_cross_mean_{suffix}",
            flow_col=f"flow_forcing_min_{suffix}",
            out_col=f"physics_index_wave_forcing_cross_mean_over_flow_forcing_min_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_cross_max_{suffix}",
            flow_col=f"flow_forcing_mean_{suffix}",
            out_col=f"physics_index_wave_forcing_cross_max_over_flow_forcing_mean_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_cross_max_{suffix}",
            flow_col=f"flow_forcing_min_{suffix}",
            out_col=f"physics_index_wave_forcing_cross_max_over_flow_forcing_min_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_along_mean_{suffix}",
            flow_col=f"flow_forcing_mean_{suffix}",
            out_col=f"physics_index_wave_forcing_along_mean_over_flow_forcing_mean_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_along_mean_{suffix}",
            flow_col=f"flow_forcing_min_{suffix}",
            out_col=f"physics_index_wave_forcing_along_mean_over_flow_forcing_min_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_along_max_{suffix}",
            flow_col=f"flow_forcing_mean_{suffix}",
            out_col=f"physics_index_wave_forcing_along_max_over_flow_forcing_mean_{suffix}",
        )
        add_physics_index_if_present(
            out,
            wave_col=f"wave_forcing_along_max_{suffix}",
            flow_col=f"flow_forcing_min_{suffix}",
            out_col=f"physics_index_wave_forcing_along_max_over_flow_forcing_min_{suffix}",
        )

    return out


# =========================
# Main feature-table builder
# =========================


def build_feature_table(
    anchors_df: pd.DataFrame,
    waves_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    streamflow_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    valid_regions: list[int],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Build the final feature table.

    Returns:
    - feature_table
    """
    if config is None:
        config = FeatureConfig()
    print("Preparing anchors...")

    anchors = prepare_anchors(anchors_df)
    print(f"Prepared anchors: {len(anchors):,} rows")
    print("Loading CDIP orientation lookup tables...")
    nearest_cdip_path = "/Volumes/x10pro/estuary/drivers_data/geos/nearest_cdip_points.csv"
    cdip_points_meta_path = "/Volumes/x10pro/estuary/drivers_data/geos/cdip_points_meta.csv"

    nearest_cdip_df = pd.read_csv(nearest_cdip_path)
    cdip_points_meta_df = pd.read_csv(cdip_points_meta_path)
    len_meta = len(cdip_points_meta_df)
    print(
        f"Loaded CDIP lookup tables: nearest={len(nearest_cdip_df):,} rows, meta={len_meta:,} rows",
    )
    print("Preparing wave orientation lookup...")
    wave_orientation_lookup = prepare_wave_orientation_lookup(
        nearest_cdip_df=nearest_cdip_df,
        cdip_points_meta_df=cdip_points_meta_df,
    )
    print(
        f"Prepared wave orientation lookup: {len(wave_orientation_lookup):,} region rows",
    )

    print("Preparing wave data...")
    waves = prepare_waves(waves_df, wave_orientation_lookup, valid_regions)
    print(f"Prepared waves: {len(waves):,} rows")

    print("Preparing tide data...")
    tide = prepare_tide(tide_df, valid_regions)
    print(f"Prepared tide: {len(tide):,} rows")

    print("Preparing streamflow data...")
    flow = prepare_streamflow(streamflow_df, valid_regions)
    print(f"Prepared streamflow: {len(flow):,} rows")

    print("Building hourly aligned wave-flow physics table...")
    physics = build_hourly_wave_flow_physics_table(waves, flow)
    print(f"Prepared hourly physics table: {len(physics):,} rows")

    print("Preparing anchor-state lookup...")
    anchor_state = build_anchor_state_lookup(predictions_df)
    print(f"Prepared anchor-state lookup: {len(anchor_state):,} rows")

    feature_frames: list[pd.DataFrame] = []

    grouped = list(anchors.groupby("region", sort=False))
    region_iter = iter_items(
        grouped,
        total=len(grouped),
        desc="Building feature table",
        enabled=config.show_progress,
    )

    for region, anchors_region in region_iter:
        print_progress(config.show_progress, f"Building features for region {region}")

        wave_region = get_region_slice(waves, region)
        tide_region = get_region_slice(tide, region)
        flow_region = get_region_slice(flow, region)
        physics_region = get_region_slice(physics, region)

        sub_features = build_features_for_region(
            anchors_region=anchors_region.reset_index(drop=True),
            wave_region=wave_region,
            tide_region=tide_region,
            flow_region=flow_region,
            physics_region=physics_region,
            config=config,
        )

        if not sub_features.empty:
            feature_frames.append(sub_features)

    if feature_frames:
        feature_table = pd.concat(feature_frames, ignore_index=True)
    else:
        feature_table = anchors.copy()

    # Merge p_open at anchor.
    feature_table = feature_table.merge(
        anchor_state,
        on=["region", "anchor_time"],
        how="left",
    )
    feature_table = add_derived_feature_columns(feature_table, config)

    # Convenience ordering.
    preferred_cols = [
        "region",
        "anchor_time",
        "horizon_days",
        "event_y",
        "censored",
        "time_to_closure_days",
        "p_open_anchor",
    ]
    remaining_cols = [c for c in feature_table.columns if c not in preferred_cols]
    feature_table = feature_table[
        [c for c in preferred_cols if c in feature_table.columns] + remaining_cols
    ]

    return feature_table


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    anchors_path = (
        "/Users/kyledorman/data/results/estuary/train/20260305-095554/"
        "merged_all_regions_timeseries_preds_hazard_anchors.csv"
    )
    predictions_path = (
        "/Users/kyledorman/data/results/estuary/train/20260305-095554/"
        "merged_all_regions_timeseries_preds.csv"
    )
    waves_path = "/Volumes/x10pro/estuary/drivers_data/cdip_waves.parquet"
    tide_path = "/Volumes/x10pro/estuary/drivers_data/fes2022_tides.parquet"
    streamflow_path = "/Volumes/x10pro/estuary/drivers_data/usgs_data_iv.parquet"

    print("Loading anchors...")
    anchors_df = pd.read_csv(anchors_path)
    print(f"Loaded anchors: {len(anchors_df):,} rows")

    valid_regions = anchors_df["region"].unique().tolist()

    print("Loading predictions...")
    predictions_df = pd.read_csv(predictions_path)
    print(f"Loaded predictions: {len(predictions_df):,} rows")
    print("Loading waves...")
    if waves_path.endswith(".parquet"):
        waves_df = pd.read_parquet(waves_path)
    else:
        waves_df = pd.read_csv(waves_path)
    print(f"Loaded waves: {len(waves_df):,} rows")
    print("Loading tide...")
    if tide_path.endswith(".parquet"):
        tide_df = pd.read_parquet(tide_path)
    else:
        tide_df = pd.read_csv(tide_path)
    print(f"Loaded tide: {len(tide_df):,} rows")
    print("Loading streamflow...")
    if streamflow_path.endswith(".parquet"):
        streamflow_df = pd.read_parquet(streamflow_path)
    else:
        streamflow_df = pd.read_csv(streamflow_path)
    print(f"Loaded streamflow: {len(streamflow_df):,} rows")
    print("Building feature table...")

    config = FeatureConfig()

    feature_table = build_feature_table(
        anchors_df=anchors_df,
        waves_df=waves_df,
        tide_df=tide_df,
        streamflow_df=streamflow_df,
        predictions_df=predictions_df,
        valid_regions=valid_regions,
        config=config,
    )
    print(
        f"Built feature table: {len(feature_table):,} rows, {len(feature_table.columns):,} columns"
    )

    if config.save_outputs:
        feature_table_path = get_output_path(anchors_path)
        feature_table.to_parquet(feature_table_path, index=False)

        print(f"Saved feature table to {feature_table_path}")

    print("Feature table preview:")
    print(feature_table.head())
