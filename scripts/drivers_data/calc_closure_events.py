from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# =========================
# Configuration
# =========================


@dataclass
class EventConfig:
    # Conservative first-pass thresholds.
    # Adjust these after you inspect the probability distributions.
    open_thresh: float = 0.9
    closed_thresh: float = 0.1

    # Maximum allowed gap between consecutive observations in a valid motif.
    max_gap_days: float = 7.0

    # Horizons for hazard labeling.
    horizons_days: tuple[int, ...] = (3, 7, 14)

    # Fraction of the horizon by which we want to see an open observation
    # to support a negative label.
    late_open_fraction: float = 0.75

    # Whether to include low-confidence closure events in hazard labeling.
    include_low_confidence_events: bool = False

    # Whether to show per-region progress bars / progress messages.
    show_progress: bool = True

    # Whether to save the output tables to disk.
    save_outputs: bool = True

    valid_sites: tuple[int, ...] = (
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


# =========================
# Basic utilities
# =========================


def validate_input_columns(df: pd.DataFrame) -> None:
    """Check that the expected columns exist."""
    required = {
        "region",
        "date",
        "p_open",
        "p_closed",
        "y_pred_unsure",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def prepare_predictions(df: pd.DataFrame, config: EventConfig) -> pd.DataFrame:
    """
    Copy the input dataframe and enforce basic typing / sorting assumptions.
    """
    validate_input_columns(df)

    out = df.copy()
    out = out[out["region"].isin(config.valid_sites)].copy()
    out = out[out["y_pred_unsure"] == 0].copy()

    # Parse datetime.
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad_n = int(out["date"].isna().sum())
        raise ValueError(f"Found {bad_n} rows with invalid date values.")

    # Sort by site and time.
    out = out.sort_values(["region", "date"]).reset_index(drop=True)

    return out


def assign_state_from_probabilities(
    p_open: float,
    open_thresh: float,
    closed_thresh: float,
) -> str:
    """
    Map p_open to a discrete confident state.
    """
    if pd.isna(p_open):
        return "uncertain"
    if p_open >= open_thresh:
        return "open"
    if p_open <= closed_thresh:
        return "closed"
    return "uncertain"


def add_confident_state(df: pd.DataFrame, config: EventConfig) -> pd.DataFrame:
    """
    Add a confident state column based on p_open thresholds.
    """
    out = df.copy()
    out["state"] = out["p_open"].apply(
        lambda x: assign_state_from_probabilities(
            p_open=x,
            open_thresh=config.open_thresh,
            closed_thresh=config.closed_thresh,
        )
    )
    return out


def add_pairwise_gap_columns(df: pd.DataFrame, config: EventConfig) -> pd.DataFrame:
    """
    Add previous/next gap columns within each region.
    """
    out = df.copy()

    out["dt_prev_days"] = out.groupby("region")["date"].diff().dt.total_seconds() / 86400.0
    out["dt_next_days"] = out.groupby("region")["date"].diff(-1).abs().dt.total_seconds() / 86400.0

    out["usable_prev"] = out["dt_prev_days"].le(config.max_gap_days).fillna(False)
    out["usable_next"] = out["dt_next_days"].le(config.max_gap_days).fillna(False)

    return out


def build_state_sequence(df: pd.DataFrame, config: EventConfig) -> pd.DataFrame:
    """
    Full preprocessing pass.
    """
    out = prepare_predictions(df, config=config)
    out = add_confident_state(out, config)
    out = add_pairwise_gap_columns(out, config)
    return out


# =========================
# Region helpers
# =========================


def days_between(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Return elapsed days between two timestamps."""
    return (t1 - t0).total_seconds() / 86400.0


def midpoint_timestamp(t0: pd.Timestamp, t1: pd.Timestamp) -> pd.Timestamp:
    """Return the midpoint between two timestamps."""
    return t0 + (t1 - t0) / 2


# =========================
# Run-length logic
# =========================


def backward_open_run(region_df: pd.DataFrame, i: int) -> int:
    """
    Count consecutive confident open observations immediately before index i.

    The closure candidate is assumed to be at index i.
    The open run ends at i - 1.
    Pairwise gap constraints are enforced.
    """
    run = 0
    j = i - 1

    while j >= 0:
        if region_df.at[j, "state"] != "open":
            break

        # Immediate open -> closed transition must be a valid pair.
        if j == i - 1 and not bool(region_df.at[i, "usable_prev"]):
            break

        # For earlier open observations in the run, the link to the next row
        # in the run must also be valid.
        if j < i - 1 and not bool(region_df.at[j + 1, "usable_prev"]):
            break

        run += 1
        j -= 1

    return run


def forward_closed_run(region_df: pd.DataFrame, i: int) -> int:
    """
    Count consecutive confident closed observations starting at index i.

    Pairwise gap constraints are enforced.
    """
    run = 0
    j = i

    while j < len(region_df):
        if region_df.at[j, "state"] != "closed":
            break

        if j > i and not bool(region_df.at[j, "usable_prev"]):
            break

        run += 1
        j += 1

    return run


def local_pattern_string(
    region_df: pd.DataFrame,
    i: int,
    n_back: int = 2,
    n_forward: int = 2,
) -> str:
    """
    Build a short local state pattern string around index i.

    This is mainly for QA and motif labeling.
    """
    back_states: list[str] = []
    j = i - 1

    while j >= 0 and len(back_states) < n_back:
        state = str(region_df.at[j, "state"])
        if state == "uncertain":
            break

        if j == i - 1 and not bool(region_df.at[i, "usable_prev"]):
            break

        if j < i - 1 and not bool(region_df.at[j + 1, "usable_prev"]):
            break

        back_states.append(state)
        j -= 1

    back_states.reverse()

    forward_states: list[str] = []
    j = i
    while j < len(region_df) and len(forward_states) < n_forward:
        state = str(region_df.at[j, "state"])
        if state == "uncertain":
            break

        if j > i and not bool(region_df.at[j, "usable_prev"]):
            break

        forward_states.append(state)
        j += 1

    return "/".join(back_states + forward_states)


# =========================
# Event scoring and motif labeling
# =========================


def compute_closure_score(
    pre_open_run: int,
    post_closed_run: int,
    transition_gap_days: float,
    max_gap_days: float,
) -> float:
    """
    Simple interpretable score.

    Persistence contributes positive evidence.
    Larger transition gaps reduce confidence slightly.
    """
    score = 0.0

    if pre_open_run >= 1:
        score += 1.0
    if pre_open_run >= 2:
        score += 1.0
    if post_closed_run >= 1:
        score += 1.0
    if post_closed_run >= 2:
        score += 1.0

    if pd.notna(transition_gap_days) and max_gap_days > 0:
        gap_penalty = transition_gap_days / max_gap_days
        score -= 0.5 * gap_penalty

    return score


def classify_closure_motif(
    pre_open_run: int,
    post_closed_run: int,
    local_pattern: str,
) -> str:
    """
    Assign a coarse motif label.

    This follows the current rules:
    - high: open/open/closed/closed
    - medium: open/closed, open/closed/closed, open/open/closed
    - low: open/open/closed/open
    - ambiguous: oscillatory examples like closed/open/closed/open
    """
    if pre_open_run >= 2 and post_closed_run >= 2:
        return "high"

    if local_pattern == "open/open/closed/open":
        return "low"

    if local_pattern == "closed/open/closed/open":
        return "ambiguous"

    if pre_open_run >= 1 and post_closed_run >= 1:
        return "medium"

    return "ambiguous"


def include_event_in_v1(motif_label: str, config: EventConfig) -> bool:
    """
    Decide whether an extracted closure event should be used for initial
    hazard labeling.
    """
    if motif_label in {"high", "medium"}:
        return True
    if motif_label == "low":
        return config.include_low_confidence_events
    return False


# =========================
# Closure event extraction
# =========================


def is_candidate_first_closed(region_df: pd.DataFrame, i: int) -> bool:
    """
    A closure candidate is the first confident closed observation after a
    confident open observation, with an acceptable gap between them.
    """
    if i <= 0:
        return False

    curr_state = region_df.at[i, "state"]
    prev_state = region_df.at[i - 1, "state"]
    usable_prev = bool(region_df.at[i, "usable_prev"])

    return curr_state == "closed" and prev_state == "open" and usable_prev


def extract_closure_events_for_region(
    region_df: pd.DataFrame,
    config: EventConfig,
) -> pd.DataFrame:
    """
    Extract one row per closure episode for a single region.
    """
    events: list[dict] = []
    region_name = int(region_df["region"].iloc[0])

    for i in range(1, len(region_df)):
        if not is_candidate_first_closed(region_df, i):
            continue

        prev_idx = i - 1

        pre_open_run = backward_open_run(region_df, i)
        post_closed_run = forward_closed_run(region_df, i)
        pattern = local_pattern_string(region_df, i, n_back=2, n_forward=2)

        last_open_time = region_df.at[prev_idx, "date"]
        first_closed_time = region_df.at[i, "date"]
        transition_gap_days = float(region_df.at[i, "dt_prev_days"])  # pyright: ignore[reportArgumentType]
        midpoint_time = midpoint_timestamp(last_open_time, first_closed_time)  # pyright: ignore[reportArgumentType]

        score = compute_closure_score(
            pre_open_run=pre_open_run,
            post_closed_run=post_closed_run,
            transition_gap_days=transition_gap_days,
            max_gap_days=config.max_gap_days,
        )
        motif = classify_closure_motif(
            pre_open_run=pre_open_run,
            post_closed_run=post_closed_run,
            local_pattern=pattern,
        )

        events.append(
            {
                "region": region_name,
                "event_index_in_region": i,
                "last_open_index_in_region": prev_idx,
                "last_open_time": last_open_time,
                "first_closed_time": first_closed_time,
                "midpoint_time": midpoint_time,
                "transition_gap_days": transition_gap_days,
                "pre_open_run": pre_open_run,
                "post_closed_run": post_closed_run,
                "local_pattern": pattern,
                "closure_score": score,
                "motif_label": motif,
                "include_v1": include_event_in_v1(motif, config),
            }
        )

    return pd.DataFrame(events)


def iter_regions(
    grouped: Iterable[tuple[object, pd.DataFrame]],
    total: int,
    desc: str,
    enabled: bool,
):
    """
    Wrap grouped region iteration in a tqdm progress bar when available.
    Fall back to plain iteration otherwise.
    """
    if enabled and tqdm is not None:
        return tqdm(grouped, total=total, desc=desc)
    return grouped


def print_progress_message(enabled: bool, message: str) -> None:
    """Print a simple progress message when progress reporting is enabled."""
    if enabled and tqdm is None:
        print(message)


def get_output_paths(input_path: Path) -> tuple[Path, Path]:
    """
    Build output file paths in the same directory as the prediction file.
    """
    stem = input_path.stem
    output_dir = input_path.parent

    closure_events_path = output_dir / f"{stem}_closure_events.csv"
    hazard_anchors_path = output_dir / f"{stem}_hazard_anchors.csv"

    return closure_events_path, hazard_anchors_path


def extract_all_closure_events(
    state_df: pd.DataFrame,
    config: EventConfig,
) -> pd.DataFrame:
    """
    Extract closure events across all regions.
    """
    event_frames: list[pd.DataFrame] = []
    grouped = list(state_df.groupby("region", sort=False))
    region_iter = iter_regions(
        grouped=grouped,
        total=len(grouped),
        desc="Extracting closure events",
        enabled=config.show_progress,
    )

    for region, region_df in region_iter:
        print_progress_message(
            config.show_progress, f"Extracting closure events for region {region}"
        )
        sub_events = extract_closure_events_for_region(
            region_df=region_df.reset_index(drop=True),
            config=config,
        )
        if not sub_events.empty:
            event_frames.append(sub_events)

    if not event_frames:
        return pd.DataFrame(
            columns=[
                "region",
                "event_index_in_region",
                "last_open_index_in_region",
                "last_open_time",
                "first_closed_time",
                "midpoint_time",
                "transition_gap_days",
                "pre_open_run",
                "post_closed_run",
                "local_pattern",
                "closure_score",
                "motif_label",
                "include_v1",
            ]
        )

    return pd.concat(event_frames, ignore_index=True)


# =========================
# Hazard labeling helpers
# =========================


def get_next_closure_event_after_time(
    region_events: pd.DataFrame,
    anchor_time: pd.Timestamp,
    t_end: pd.Timestamp,
) -> pd.Series | None:
    """
    Return the first included closure event strictly after anchor_time.
    """
    candidates = region_events.loc[
        (region_events["include_v1"])
        & (region_events["first_closed_time"] > anchor_time)
        & (region_events["first_closed_time"] <= t_end)
    ].sort_values("first_closed_time")

    if candidates.empty:
        return None

    return candidates.iloc[0]


def has_adequate_followup_for_negative(
    region_df: pd.DataFrame,
    anchor_idx: int,
    horizon_days: int,
    config: EventConfig,
) -> bool:
    """
    Determine whether an open anchor can safely be labeled negative.

    Conservative first-pass rule:
    - No disqualifying large gaps in observed follow-up
    - No uncertain states in the observed follow-up window
    - No closed observations in the window
    - At least one open observation near the end of the horizon
    """
    t0: pd.Timestamp = region_df.at[anchor_idx, "date"]  # pyright: ignore[reportAssignmentType]
    t_end = t0 + pd.Timedelta(days=horizon_days)

    found_late_open = False

    for j in range(anchor_idx + 1, len(region_df)):
        t: pd.Timestamp = region_df.at[j, "date"]  # pyright: ignore[reportAssignmentType]
        if t > t_end:
            break

        if not bool(region_df.at[j, "usable_prev"]):
            return False

        state = region_df.at[j, "state"]

        if state == "uncertain":
            return False

        if state == "closed":
            return False

        if state == "open":
            frac = days_between(t0, t) / float(horizon_days)
            if frac >= config.late_open_fraction:
                found_late_open = True

    return found_late_open


def make_anchor_record(
    region: int,
    anchor_idx: int,
    anchor_time: pd.Timestamp,
    horizon_days: int,
    event_y: int | None,
    censored: bool,
    linked_event: pd.Series | None,
) -> dict:
    """
    Build one hazard-anchor output row.
    """
    record = {
        "region": region,
        "anchor_index_in_region": anchor_idx,
        "anchor_time": anchor_time,
        "horizon_days": horizon_days,
        "event_y": event_y,
        "censored": censored,
        "time_to_closure_days": None,
        "linked_event_time": None,
        "linked_event_midpoint_time": None,
        "linked_closure_score": None,
        "linked_motif_label": None,
    }

    if linked_event is not None:
        record["linked_event_time"] = linked_event["first_closed_time"]
        record["linked_event_midpoint_time"] = linked_event["midpoint_time"]
        record["linked_closure_score"] = linked_event["closure_score"]
        record["linked_motif_label"] = linked_event["motif_label"]

        if event_y == 1:
            record["time_to_closure_days"] = days_between(
                anchor_time,
                linked_event["first_closed_time"],
            )

    return record


# =========================
# Hazard anchor labeling
# =========================


def build_hazard_anchors_for_region(
    region_df: pd.DataFrame,
    region_events: pd.DataFrame,
    config: EventConfig,
) -> pd.DataFrame:
    """
    Create one row per open anchor per horizon.

    Multiple anchors may map to the same closure event. That is intentional.
    """
    anchors: list[dict] = []
    region_name = int(region_df["region"].iloc[0])

    for i in range(len(region_df)):
        if region_df.at[i, "state"] != "open":
            continue

        anchor_time: pd.Timestamp = region_df.at[i, "date"]  # pyright: ignore[reportAssignmentType]

        for horizon_days in config.horizons_days:
            t_end = anchor_time + pd.Timedelta(days=horizon_days)
            next_event = get_next_closure_event_after_time(region_events, anchor_time, t_end)

            # Positive if the next included closure event occurs within the horizon.
            if next_event is not None:
                anchors.append(
                    make_anchor_record(
                        region=region_name,
                        anchor_idx=i,
                        anchor_time=anchor_time,
                        horizon_days=horizon_days,
                        event_y=1,
                        censored=False,
                        linked_event=next_event,
                    )
                )
                continue

            # Otherwise, it is either a negative or censored.
            adequate = has_adequate_followup_for_negative(
                region_df=region_df,
                anchor_idx=i,
                horizon_days=horizon_days,
                config=config,
            )

            if adequate:
                anchors.append(
                    make_anchor_record(
                        region=region_name,
                        anchor_idx=i,
                        anchor_time=anchor_time,
                        horizon_days=horizon_days,
                        event_y=0,
                        censored=False,
                        linked_event=None,
                    )
                )
            else:
                anchors.append(
                    make_anchor_record(
                        region=region_name,
                        anchor_idx=i,
                        anchor_time=anchor_time,
                        horizon_days=horizon_days,
                        event_y=None,
                        censored=True,
                        linked_event=None,
                    )
                )

    return pd.DataFrame(anchors)


def build_all_hazard_anchors(
    state_df: pd.DataFrame,
    closure_events: pd.DataFrame,
    config: EventConfig,
) -> pd.DataFrame:
    """
    Build hazard labels across all regions.
    """
    anchor_frames: list[pd.DataFrame] = []
    grouped = list(state_df.groupby("region", sort=False))
    region_iter = iter_regions(
        grouped=grouped,
        total=len(grouped),
        desc="Building hazard anchors",
        enabled=config.show_progress,
    )

    for region, region_df in region_iter:
        print_progress_message(config.show_progress, f"Building hazard anchors for region {region}")
        region_df = region_df.reset_index(drop=True)

        region_events = (
            closure_events.loc[closure_events["region"] == region]
            .sort_values("first_closed_time")
            .reset_index(drop=True)
        )

        sub_anchors = build_hazard_anchors_for_region(
            region_df=region_df,
            region_events=region_events,
            config=config,
        )
        if not sub_anchors.empty:
            anchor_frames.append(sub_anchors)

    if not anchor_frames:
        return pd.DataFrame(
            columns=[
                "region",
                "anchor_index_in_region",
                "anchor_time",
                "horizon_days",
                "event_y",
                "censored",
                "time_to_closure_days",
                "linked_event_time",
                "linked_event_midpoint_time",
                "linked_closure_score",
                "linked_motif_label",
            ]
        )

    return pd.concat(anchor_frames, ignore_index=True)


# =========================
# Convenience wrapper
# =========================


def build_closure_event_and_anchor_tables(
    df: pd.DataFrame,
    config: EventConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end convenience wrapper.

    Returns:
    - state_df
    - closure_events
    - hazard_anchors
    """
    if config is None:
        config = EventConfig()

    state_df = build_state_sequence(df, config)
    closure_events = extract_all_closure_events(state_df, config)
    hazard_anchors = build_all_hazard_anchors(
        state_df,
        closure_events,
        config,
    )

    return state_df, closure_events, hazard_anchors


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    pred_path = Path(
        "/Users/kyledorman/data/results/estuary/train/20260305-095554/merged_all_regions_timeseries_preds.csv"
    )
    pred_df = pd.read_csv(pred_path)

    config = EventConfig()  # valid_sites=(15,), save_outputs=False)

    state_df, closure_events, hazard_anchors = build_closure_event_and_anchor_tables(
        df=pred_df,
        config=config,
    )

    if config.save_outputs:
        closure_events_path, hazard_anchors_path = get_output_paths(pred_path)
        closure_events.to_csv(closure_events_path, index=False)
        hazard_anchors.to_csv(hazard_anchors_path, index=False)
        print(f"Saved closure events to {closure_events_path}")
        print(f"Saved hazard anchors to {hazard_anchors_path}")

    print("State sequence preview:")
    print(state_df.head())

    print("\nClosure events:")
    print(closure_events.head())

    print("\nHazard anchors:")
    print(hazard_anchors.head())
