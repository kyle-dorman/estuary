from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd

DEFAULT_DRIVERS_DIR = Path(
    "/Users/kyledorman/data/results/estuary/train/20260305-095554/seasonal_drivers"
)

DAILY_DRIVER_TABLE = "daily_driver_table.parquet"
TRAIN_EVAL_DATASET = "train_eval_pair_dataset.parquet"
EVENT_WINDOWS_TABLE = "transition_event_windows.parquet"
LABELED_TIMELINE_TABLE = "labeled_timeline.parquet"
MODEL_TIMELINE_TABLE = "transition_model_timeline.parquet"

DEFAULT_SKIP_REGIONS = (12103,)
DEFAULT_ROLLING_WINDOW_DAYS = (3, 5, 7)
DEFAULT_ROLLING_FEATURE_DAYS = 7
DEFAULT_TIMELINE_FEATURE_LAG_DAYS = 1
DEFAULT_PAIR_FEATURE_LAG_DAYS = 7
DEFAULT_MIN_OPEN = 0.7
DEFAULT_MIN_CLOSED = 0.3
DEFAULT_MAX_UNCERTAIN_BRIDGE_GAP_DAYS = 7
DEFAULT_MAX_OBS_GAP_DAYS = 14

REGION_COL = "region"
DATE_COL = "date"
STATE_OPEN = "open"
STATE_CLOSED = "closed"
STATE_UNCERTAIN = "uncertain"

BASE_FEATURE_COLS = [
    "flow_daily",
    "flow_window",
    "wave_daily",
    "wave_window",
    "tide_daily",
    "tide_window",
    "month_sin",
    "month_cos",
]

REQUIRED_DAILY_COLUMNS = {
    REGION_COL,
    DATE_COL,
    "p_open",
    "is_obs",
    "streamflow_mean",
    "wave_energy_cross_max",
    "wave_energy_cross_max_rank_norm",
    "tide_range",
    "tide_range_rank_norm",
}

EVENT_WINDOW_COLUMNS = [
    REGION_COL,
    "window_start_date",
    "window_end_date",
    "obs_gap_days",
    "window_start_state",
    "window_end_state",
    "window_start_p_open",
    "window_end_p_open",
    "transition",
    "opening_eval_window",
    "opening_change",
    "closing_eval_window",
    "closing_change",
]

LABELED_SOURCE_COLUMNS = [
    REGION_COL,
    DATE_COL,
    "pair_state",
    "source_window_type",
    "source_window_start_date",
    "source_window_end_date",
    "source_obs_gap_days",
]


@dataclass(frozen=True)
class RollingFeature:
    name: str
    source_col: str
    agg: str


ROLLING_FEATURES = (
    RollingFeature("flow_sum", "streamflow_mean", "sum"),
    RollingFeature("wave_sum", "wave_energy_cross_max", "sum"),
    RollingFeature("tide_mean", "tide_range", "mean"),
)


@dataclass(frozen=True)
class TrainEvalConfig:
    drivers_dir: Path = DEFAULT_DRIVERS_DIR
    output_dir: Path | None = None
    skip_regions: tuple[int, ...] = DEFAULT_SKIP_REGIONS
    rolling_window_days: tuple[int, ...] = DEFAULT_ROLLING_WINDOW_DAYS
    rolling_feature_days: int = DEFAULT_ROLLING_FEATURE_DAYS
    timeline_feature_lag_days: int = DEFAULT_TIMELINE_FEATURE_LAG_DAYS
    pair_feature_lag_days: int = DEFAULT_PAIR_FEATURE_LAG_DAYS
    min_open: float = DEFAULT_MIN_OPEN
    min_closed: float = DEFAULT_MIN_CLOSED
    max_uncertain_bridge_gap_days: int = DEFAULT_MAX_UNCERTAIN_BRIDGE_GAP_DAYS
    max_obs_gap_days: int = DEFAULT_MAX_OBS_GAP_DAYS
    write_intermediates: bool = True

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or self.drivers_dir


@dataclass(frozen=True)
class TrainEvalFrames:
    model_timeline: pd.DataFrame
    event_windows: pd.DataFrame
    labeled_timeline: pd.DataFrame
    train_eval_pairs: pd.DataFrame
    feature_cols: list[str]


def validate_config(config: TrainEvalConfig) -> None:
    if config.min_closed >= config.min_open:
        raise click.ClickException("--min-closed must be less than --min-open")

    positive_day_fields = {
        "--rolling-feature-days": config.rolling_feature_days,
        "--timeline-feature-lag-days": config.timeline_feature_lag_days,
        "--pair-feature-lag-days": config.pair_feature_lag_days,
        "--max-uncertain-bridge-gap-days": config.max_uncertain_bridge_gap_days,
        "--max-obs-gap-days": config.max_obs_gap_days,
    }
    for option, value in positive_day_fields.items():
        if value < 1:
            raise click.ClickException(f"{option} must be >= 1")

    if any(days < 1 for days in config.rolling_window_days):
        raise click.ClickException("--rolling-window-days values must be >= 1")


def read_daily_driver_table(drivers_dir: Path, skip_regions: Sequence[int]) -> pd.DataFrame:
    path = drivers_dir / DAILY_DRIVER_TABLE
    daily = pd.read_parquet(path)
    validate_columns(daily, REQUIRED_DAILY_COLUMNS, path.name)

    if skip_regions:
        daily = daily[~daily[REGION_COL].isin(skip_regions)].copy()
    else:
        daily = daily.copy()

    daily[DATE_COL] = pd.to_datetime(daily[DATE_COL])
    return daily.sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)


def validate_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        raise click.ClickException(f"{table_name} is missing required columns: {sorted(missing)}")


def rolling_window_days(config: TrainEvalConfig) -> tuple[int, ...]:
    return tuple(sorted({*config.rolling_window_days, config.rolling_feature_days}))


def time_rolling_by_region(
    df: pd.DataFrame,
    value_col: str,
    window_days: int,
    agg: str,
    min_periods: int = 1,
) -> pd.Series:
    window = f"{window_days}D"
    result = pd.Series(np.nan, index=df.index, dtype=float)

    for _, group in df.sort_values([REGION_COL, DATE_COL]).groupby(REGION_COL, sort=False):
        rolled = (
            group.set_index(DATE_COL)[value_col]
            .rolling(window=window, min_periods=min_periods, center=False)
            .agg(agg)
        )
        result.loc[group.index] = rolled.to_numpy()

    return result


def add_rolling_driver_features(
    daily: pd.DataFrame,
    window_days: Sequence[int],
) -> pd.DataFrame:
    out = daily.copy()

    for days in window_days:
        for feature in ROLLING_FEATURES:
            out[f"{feature.name}_rolling_{days}d"] = time_rolling_by_region(
                out,
                value_col=feature.source_col,
                window_days=days,
                agg=feature.agg,
                min_periods=1,
            )

    return out


def rank_norm(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def zero_aware_rank_norm(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)

    is_zero = s == 0
    is_pos = s > 0

    out[is_zero] = 0.0
    out[is_pos] = s[is_pos].rank(pct=True)

    return out


def state_from_probability(
    p_open: pd.Series,
    min_open: float,
    min_closed: float,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                p_open >= min_open,
                p_open <= min_closed,
            ],
            [
                STATE_OPEN,
                STATE_CLOSED,
            ],
            default=STATE_UNCERTAIN,
        ),
        index=p_open.index,
    )


def add_lagged_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_lag_days: int,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    lagged_feature_cols = []
    key_cols = [REGION_COL, DATE_COL]

    for feature_col in feature_cols:
        lagged_col = f"{feature_col}_lag{feature_lag_days}"
        lag_source = out[[REGION_COL, DATE_COL, feature_col]].copy()
        lag_source[DATE_COL] = lag_source[DATE_COL] + pd.Timedelta(days=feature_lag_days)  # pyright: ignore[reportOperatorIssue]
        lag_source = lag_source.rename(columns={feature_col: lagged_col})
        out = out.merge(lag_source, on=key_cols, how="left")
        lagged_feature_cols.append(lagged_col)

    return out, lagged_feature_cols


def add_month_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out[DATE_COL].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def build_transition_model_frame(
    daily: pd.DataFrame,
    rolling_feature_days: int,
    timeline_feature_lag_days: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    validate_columns(
        daily,
        {
            f"flow_sum_rolling_{rolling_feature_days}d",
            f"wave_sum_rolling_{rolling_feature_days}d",
            f"tide_mean_rolling_{rolling_feature_days}d",
        },
        DAILY_DRIVER_TABLE,
    )

    model_df = daily.copy().sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)

    model_df["flow_daily"] = model_df["streamflow_mean_rank_norm"]
    model_df["wave_daily"] = model_df["wave_energy_cross_max_rank_norm"]
    model_df["tide_daily"] = model_df["tide_range_rank_norm"]

    model_df["flow_window"] = model_df.groupby(REGION_COL)[
        f"flow_sum_rolling_{rolling_feature_days}d"
    ].transform(zero_aware_rank_norm)
    model_df["wave_window"] = model_df.groupby(REGION_COL)[
        f"wave_sum_rolling_{rolling_feature_days}d"
    ].transform(rank_norm)
    model_df["tide_window"] = model_df.groupby(REGION_COL)[
        f"tide_mean_rolling_{rolling_feature_days}d"
    ].transform(rank_norm)

    current_feature_cols = BASE_FEATURE_COLS[:-2]
    model_df, lagged_feature_cols = add_lagged_feature_columns(
        model_df,
        feature_cols=current_feature_cols,
        feature_lag_days=timeline_feature_lag_days,
    )

    model_df = add_month_features(model_df)

    timeline_feature_cols = lagged_feature_cols + ["month_sin", "month_cos"]
    timeline_summary_cols = [
        REGION_COL,
        DATE_COL,
        "p_open",
        *timeline_feature_cols,
    ]

    return model_df, timeline_feature_cols, timeline_summary_cols


def resolve_uncertain_bridges(
    obs_df: pd.DataFrame,
    max_uncertain_bridge_gap_days: int,
) -> pd.DataFrame:
    out = obs_df.copy()
    out["obs_state"] = out["obs_state_raw"].copy()
    out["drop_uncertain_bridge"] = False

    for _, sub in out.groupby(REGION_COL, sort=False):
        sub = sub.sort_values(DATE_COL).copy()
        states = sub["obs_state_raw"].to_numpy()
        dates = pd.to_datetime(sub[DATE_COL]).to_numpy()
        idx = sub.index.to_numpy()

        for pos in np.where(states == STATE_UNCERTAIN)[0]:
            prev_pos = previous_observed_state_position(states, pos)
            next_pos = next_observed_state_position(states, pos)

            if prev_pos is None or next_pos is None:
                continue

            gap_before = (pd.Timestamp(dates[pos]) - pd.Timestamp(dates[prev_pos])).days
            gap_after = (pd.Timestamp(dates[next_pos]) - pd.Timestamp(dates[pos])).days
            within_bridge_gap = (
                gap_before <= max_uncertain_bridge_gap_days
                and gap_after <= max_uncertain_bridge_gap_days
            )
            if not within_bridge_gap:
                continue

            prev_state = states[prev_pos]
            next_state = states[next_pos]
            if prev_state == next_state:
                out.loc[idx[pos], "obs_state"] = prev_state
            else:
                out.loc[idx[pos], "drop_uncertain_bridge"] = True

    return out


def previous_observed_state_position(states: np.ndarray, pos: int) -> int | None:
    for candidate_pos in range(pos - 1, -1, -1):
        if states[candidate_pos] in [STATE_OPEN, STATE_CLOSED]:
            return candidate_pos
    return None


def next_observed_state_position(states: np.ndarray, pos: int) -> int | None:
    for candidate_pos in range(pos + 1, len(states)):
        if states[candidate_pos] in [STATE_OPEN, STATE_CLOSED]:
            return candidate_pos
    return None


def build_observed_state_table(
    model_df: pd.DataFrame,
    min_open: float,
    min_closed: float,
    max_uncertain_bridge_gap_days: int,
) -> pd.DataFrame:
    obs_df = model_df[model_df["is_obs"]].copy()
    obs_df = obs_df.sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)
    obs_df["obs_state_raw"] = state_from_probability(obs_df["p_open"], min_open, min_closed)
    return resolve_uncertain_bridges(obs_df, max_uncertain_bridge_gap_days)


def build_event_windows(obs_df: pd.DataFrame, max_obs_gap_days: int) -> pd.DataFrame:
    obs_clean = obs_df[~obs_df["drop_uncertain_bridge"]].copy()
    obs_clean = obs_clean.sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)

    obs_clean["prev_obs_state"] = obs_clean.groupby(REGION_COL)["obs_state"].shift(1)
    obs_clean["prev_obs_state_raw"] = obs_clean.groupby(REGION_COL)["obs_state_raw"].shift(1)
    obs_clean["prev_obs_date"] = obs_clean.groupby(REGION_COL)[DATE_COL].shift(1)
    obs_clean["prev_obs_p_open"] = obs_clean.groupby(REGION_COL)["p_open"].shift(1)
    obs_clean["obs_gap_days"] = (obs_clean[DATE_COL] - obs_clean["prev_obs_date"]).dt.days

    event_windows = obs_clean[
        obs_clean["prev_obs_state"].isin([STATE_OPEN, STATE_CLOSED])
        & obs_clean["obs_state"].isin([STATE_OPEN, STATE_CLOSED])
        & obs_clean["obs_gap_days"].notna()
        & (obs_clean["obs_gap_days"] <= max_obs_gap_days)
    ].copy()

    event_windows["transition"] = (
        event_windows["prev_obs_state"] + "_to_" + event_windows["obs_state"]
    )
    event_windows["opening_change"] = (
        event_windows["transition"] == f"{STATE_CLOSED}_to_{STATE_OPEN}"
    ).astype(int)
    event_windows["closing_change"] = (
        event_windows["transition"] == f"{STATE_OPEN}_to_{STATE_CLOSED}"
    ).astype(int)
    event_windows["opening_eval_window"] = (event_windows["prev_obs_state"] == STATE_CLOSED).astype(
        int
    )
    event_windows["closing_eval_window"] = (event_windows["prev_obs_state"] == STATE_OPEN).astype(
        int
    )

    event_windows = event_windows.rename(
        columns={
            "prev_obs_date": "window_start_date",
            DATE_COL: "window_end_date",
            "prev_obs_state": "window_start_state",
            "obs_state": "window_end_state",
            "prev_obs_p_open": "window_start_p_open",
            "p_open": "window_end_p_open",
        }
    )

    return event_windows[EVENT_WINDOW_COLUMNS].copy()


def add_window_type(event_windows: pd.DataFrame, max_obs_gap_days: int) -> pd.DataFrame:
    windows = event_windows[
        event_windows["window_start_state"].isin([STATE_OPEN, STATE_CLOSED])
        & event_windows["window_end_state"].isin([STATE_OPEN, STATE_CLOSED])
        & (event_windows["obs_gap_days"] <= max_obs_gap_days)
    ].copy()

    windows["window_type"] = np.select(
        [
            windows["transition"] == f"{STATE_CLOSED}_to_{STATE_CLOSED}",
            windows["transition"] == f"{STATE_OPEN}_to_{STATE_OPEN}",
            windows["transition"] == f"{STATE_CLOSED}_to_{STATE_OPEN}",
            windows["transition"] == f"{STATE_OPEN}_to_{STATE_CLOSED}",
        ],
        [
            "stable_closed",
            "stable_open",
            "opening_transition",
            "closing_transition",
        ],
        default="other",
    )

    return windows[windows["window_type"] != "other"].copy()


def label_rows_for_window(region_timeline: pd.DataFrame, window: pd.Series) -> pd.DataFrame:
    start_date = pd.Timestamp(window["window_start_date"])
    end_date = pd.Timestamp(window["window_end_date"])
    window_type = window["window_type"]

    if window_type == "stable_closed":
        rows = slice_timeline(region_timeline, start_date, end_date)
        rows["pair_state"] = STATE_CLOSED
    elif window_type == "stable_open":
        rows = slice_timeline(region_timeline, start_date, end_date)
        rows["pair_state"] = STATE_OPEN
    elif window_type == "opening_transition":
        rows = endpoint_rows(region_timeline, start_date, end_date)
        rows["pair_state"] = np.where(rows[DATE_COL] == start_date, STATE_CLOSED, STATE_OPEN)
    elif window_type == "closing_transition":
        rows = endpoint_rows(region_timeline, start_date, end_date)
        rows["pair_state"] = np.where(rows[DATE_COL] == start_date, STATE_OPEN, STATE_CLOSED)
    else:
        return pd.DataFrame()

    rows = rows.dropna(subset=["pair_state"]).copy()
    if rows.empty:
        return rows

    rows["source_window_type"] = window_type
    rows["source_window_start_date"] = start_date
    rows["source_window_end_date"] = end_date
    rows["source_obs_gap_days"] = window["obs_gap_days"]
    return rows


def slice_timeline(
    region_timeline: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    return region_timeline[
        (region_timeline[DATE_COL] >= start_date) & (region_timeline[DATE_COL] <= end_date)
    ].copy()


def endpoint_rows(
    region_timeline: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    return region_timeline[region_timeline[DATE_COL].isin([start_date, end_date])].copy()


def build_labeled_timeline(
    timeline: pd.DataFrame,
    event_windows: pd.DataFrame,
    max_obs_gap_days: int,
) -> pd.DataFrame:
    windows = add_window_type(event_windows, max_obs_gap_days)
    timeline = timeline.sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)
    labeled_parts = []

    for region, region_windows in windows.groupby(REGION_COL, sort=False):
        region_timeline = timeline[timeline[REGION_COL] == region].copy()
        if region_timeline.empty:
            continue

        for _, window in region_windows.iterrows():
            rows = label_rows_for_window(region_timeline, window)
            if not rows.empty:
                labeled_parts.append(rows)

    if not labeled_parts:
        return empty_labeled_timeline(timeline)

    labeled_timeline = pd.concat(labeled_parts, ignore_index=True)
    labeled_timeline = drop_conflicting_pair_states(labeled_timeline)
    return (
        labeled_timeline.sort_values([REGION_COL, DATE_COL])
        .drop_duplicates([REGION_COL, DATE_COL, "pair_state"])
        .drop_duplicates([REGION_COL, DATE_COL])
        .reset_index(drop=True)
    )


def empty_labeled_timeline(timeline: pd.DataFrame) -> pd.DataFrame:
    columns = list(timeline.columns)
    for col in LABELED_SOURCE_COLUMNS:
        if col not in columns:
            columns.append(col)
    return pd.DataFrame(columns=columns)


def drop_conflicting_pair_states(labeled_timeline: pd.DataFrame) -> pd.DataFrame:
    state_counts = (
        labeled_timeline.groupby([REGION_COL, DATE_COL])["pair_state"]
        .nunique()
        .reset_index(name="n_states")
    )
    conflict_keys = state_counts[state_counts["n_states"] > 1][[REGION_COL, DATE_COL]]

    if conflict_keys.empty:
        return labeled_timeline

    out = labeled_timeline.merge(
        conflict_keys.assign(conflicting_pair_state=True),
        on=[REGION_COL, DATE_COL],
        how="left",
    )
    return out[out["conflicting_pair_state"].isna()].drop(columns=["conflicting_pair_state"])


def build_train_eval_pair_dataset(
    labeled_timeline: pd.DataFrame,
    timeline: pd.DataFrame,
    feature_lag_days: int,
    feature_cols_base: Sequence[str] = BASE_FEATURE_COLS,
) -> tuple[pd.DataFrame, list[str]]:
    if labeled_timeline.empty:
        return empty_train_eval_pair_dataset(feature_cols_base), feature_pair_columns(
            feature_cols_base
        )

    label_state = labeled_timeline[LABELED_SOURCE_COLUMNS].copy()
    label_state = label_state.rename(
        columns={
            DATE_COL: "label_date",
            "pair_state": "state",
            "source_window_type": "label_window_type",
            "source_window_start_date": "label_window_start_date",
            "source_window_end_date": "label_window_end_date",
            "source_obs_gap_days": "label_obs_gap_days",
        }
    )
    label_state["feature_date"] = label_state["label_date"] - pd.Timedelta(days=feature_lag_days)

    feature_state = labeled_timeline[LABELED_SOURCE_COLUMNS].copy()
    feature_state = feature_state.rename(
        columns={
            DATE_COL: "feature_date",
            "pair_state": "state_prev",
            "source_window_type": "feature_window_type",
            "source_window_start_date": "feature_window_start_date",
            "source_window_end_date": "feature_window_end_date",
            "source_obs_gap_days": "feature_obs_gap_days",
        }
    )

    pair_rows = label_state.merge(feature_state, on=[REGION_COL, "feature_date"], how="inner")
    pair_rows["feature_lag_days"] = feature_lag_days
    pair_rows["is_open"] = (pair_rows["state"] == STATE_OPEN).astype(int)
    pair_rows["transition"] = transition_labels(pair_rows)

    feature_source = timeline[[REGION_COL, DATE_COL, *feature_cols_base]].copy()
    feature_source = feature_source.rename(
        columns={
            DATE_COL: "feature_date",
            **{col: f"{col}_feature" for col in feature_cols_base},
        }
    )

    train_eval_pairs = pair_rows.merge(feature_source, on=[REGION_COL, "feature_date"], how="inner")
    feature_cols = feature_pair_columns(feature_cols_base)
    train_eval_pairs = train_eval_pairs.dropna(
        subset=[
            REGION_COL,
            "label_date",
            "feature_date",
            "state",
            "state_prev",
            "is_open",
            *feature_cols,
        ]
    ).copy()
    train_eval_pairs = train_eval_pairs.sort_values([REGION_COL, "label_date"]).reset_index(
        drop=True
    )

    return train_eval_pairs, feature_cols


def transition_labels(pair_rows: pd.DataFrame) -> pd.Series:
    labels = pd.Series("persistence", index=pair_rows.index)
    labels.loc[(pair_rows["state_prev"] == STATE_CLOSED) & (pair_rows["state"] == STATE_OPEN)] = (
        "opening"
    )
    labels.loc[(pair_rows["state_prev"] == STATE_OPEN) & (pair_rows["state"] == STATE_CLOSED)] = (
        "closing"
    )
    return labels


def feature_pair_columns(feature_cols_base: Sequence[str]) -> list[str]:
    return [f"{col}_feature" for col in feature_cols_base]


def empty_train_eval_pair_dataset(feature_cols_base: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            REGION_COL,
            "label_date",
            "state",
            "label_window_type",
            "label_window_start_date",
            "label_window_end_date",
            "label_obs_gap_days",
            "feature_date",
            "state_prev",
            "feature_window_type",
            "feature_window_start_date",
            "feature_window_end_date",
            "feature_obs_gap_days",
            "feature_lag_days",
            "is_open",
            "transition",
            *feature_pair_columns(feature_cols_base),
        ]
    )


def build_train_eval_frames(config: TrainEvalConfig) -> TrainEvalFrames:
    daily = read_daily_driver_table(config.drivers_dir, config.skip_regions)
    daily = add_rolling_driver_features(daily, rolling_window_days(config))
    model_timeline, _, _ = build_transition_model_frame(
        daily,
        rolling_feature_days=config.rolling_feature_days,
        timeline_feature_lag_days=config.timeline_feature_lag_days,
    )
    observed_states = build_observed_state_table(
        model_timeline,
        min_open=config.min_open,
        min_closed=config.min_closed,
        max_uncertain_bridge_gap_days=config.max_uncertain_bridge_gap_days,
    )
    event_windows = build_event_windows(observed_states, config.max_obs_gap_days)
    labeled_timeline = build_labeled_timeline(
        model_timeline,
        event_windows,
        max_obs_gap_days=config.max_obs_gap_days,
    )
    train_eval_pairs, feature_cols = build_train_eval_pair_dataset(
        labeled_timeline,
        model_timeline,
        feature_lag_days=config.pair_feature_lag_days,
    )

    return TrainEvalFrames(
        model_timeline=model_timeline,
        event_windows=event_windows,
        labeled_timeline=labeled_timeline,
        train_eval_pairs=train_eval_pairs,
        feature_cols=feature_cols,
    )


def write_outputs(frames: TrainEvalFrames, output_dir: Path, write_intermediates: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames.train_eval_pairs.to_parquet(output_dir / TRAIN_EVAL_DATASET, index=False)

    if not write_intermediates:
        return

    frames.event_windows.to_parquet(output_dir / EVENT_WINDOWS_TABLE, index=False)
    frames.labeled_timeline.to_parquet(output_dir / LABELED_TIMELINE_TABLE, index=False)
    frames.model_timeline.to_parquet(output_dir / MODEL_TIMELINE_TABLE, index=False)


def print_summary(frames: TrainEvalFrames, output_dir: Path, write_intermediates: bool) -> None:
    click.echo(f"train/eval pair rows: {len(frames.train_eval_pairs):,}")
    click.echo(f"event windows: {len(frames.event_windows):,}")
    click.echo(f"labeled timeline rows: {len(frames.labeled_timeline):,}")

    if not frames.event_windows.empty:
        click.echo("\nEvent window transitions:")
        click.echo(frames.event_windows["transition"].value_counts().to_string())
        click.echo("\nObserved gap days by transition:")
        click.echo(
            frames.event_windows.groupby("transition")["obs_gap_days"]
            .describe()
            .round(1)
            .to_string()
        )

    if not frames.train_eval_pairs.empty:
        click.echo("\nPair transitions:")
        click.echo(frames.train_eval_pairs["transition"].value_counts().to_string())
        click.echo("\nState cross-tab:")
        click.echo(
            pd.crosstab(frames.train_eval_pairs["state_prev"], frames.train_eval_pairs["state"])
        )
        click.echo("\nSource window type transitions:")
        click.echo(
            frames.train_eval_pairs.groupby(["feature_window_type", "label_window_type"])[
                "transition"
            ]
            .value_counts()
            .to_string()
        )

    click.echo(f"\nSaved {TRAIN_EVAL_DATASET} to {output_dir}")
    if write_intermediates:
        click.echo(f"Saved {EVENT_WINDOWS_TABLE} to {output_dir}")
        click.echo(f"Saved {LABELED_TIMELINE_TABLE} to {output_dir}")
        click.echo(f"Saved {MODEL_TIMELINE_TABLE} to {output_dir}")


@click.command()
@click.option(
    "--drivers-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_DRIVERS_DIR,
    show_default=True,
    help="Directory containing daily_driver_table.parquet.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for generated datasets. Defaults to --drivers-dir.",
)
@click.option(
    "--skip-region",
    "skip_regions",
    type=int,
    multiple=True,
    default=DEFAULT_SKIP_REGIONS,
    show_default=True,
    help="Region id to exclude. Repeat the option to exclude multiple regions.",
)
@click.option(
    "--rolling-window-days",
    "rolling_window_days",
    type=int,
    multiple=True,
    default=DEFAULT_ROLLING_WINDOW_DAYS,
    show_default=True,
    help="Rolling window lengths to compute. Repeat the option for multiple windows.",
)
@click.option(
    "--rolling-feature-days",
    type=int,
    default=DEFAULT_ROLLING_FEATURE_DAYS,
    show_default=True,
    help="Rolling window length used by the model features.",
)
@click.option(
    "--timeline-feature-lag-days",
    type=int,
    default=DEFAULT_TIMELINE_FEATURE_LAG_DAYS,
    show_default=True,
    help="Lag used for timeline diagnostic feature columns.",
)
@click.option(
    "--pair-feature-lag-days",
    type=int,
    default=DEFAULT_PAIR_FEATURE_LAG_DAYS,
    show_default=True,
    help="Days between the feature date and label date in the train/eval pairs.",
)
@click.option("--min-open", type=float, default=DEFAULT_MIN_OPEN, show_default=True)
@click.option("--min-closed", type=float, default=DEFAULT_MIN_CLOSED, show_default=True)
@click.option(
    "--max-uncertain-bridge-gap-days",
    type=int,
    default=DEFAULT_MAX_UNCERTAIN_BRIDGE_GAP_DAYS,
    show_default=True,
)
@click.option(
    "--max-obs-gap-days",
    type=int,
    default=DEFAULT_MAX_OBS_GAP_DAYS,
    show_default=True,
)
@click.option(
    "--write-intermediates/--no-write-intermediates",
    default=True,
    show_default=True,
    help="Write transition windows, labeled timeline, and model timeline parquet files.",
)
def main(
    drivers_dir: Path,
    output_dir: Path | None,
    skip_regions: tuple[int, ...],
    rolling_window_days: tuple[int, ...],
    rolling_feature_days: int,
    timeline_feature_lag_days: int,
    pair_feature_lag_days: int,
    min_open: float,
    min_closed: float,
    max_uncertain_bridge_gap_days: int,
    max_obs_gap_days: int,
    write_intermediates: bool,
) -> None:
    config = TrainEvalConfig(
        drivers_dir=drivers_dir,
        output_dir=output_dir,
        skip_regions=skip_regions,
        rolling_window_days=rolling_window_days,
        rolling_feature_days=rolling_feature_days,
        timeline_feature_lag_days=timeline_feature_lag_days,
        pair_feature_lag_days=pair_feature_lag_days,
        min_open=min_open,
        min_closed=min_closed,
        max_uncertain_bridge_gap_days=max_uncertain_bridge_gap_days,
        max_obs_gap_days=max_obs_gap_days,
        write_intermediates=write_intermediates,
    )
    validate_config(config)

    click.echo(f"Loading daily drivers from {config.drivers_dir / DAILY_DRIVER_TABLE}")
    frames = build_train_eval_frames(config)
    write_outputs(frames, config.resolved_output_dir, config.write_intermediates)
    print_summary(frames, config.resolved_output_dir, config.write_intermediates)


if __name__ == "__main__":
    main()
