from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from estuary.model.data import parse_dt_from_pth


def hysteresis_decode(
    times: pd.Series,  # datetime-like, sorted
    probs: np.ndarray,  # p(open) in [0,1], same length as times
    T_high: float = 0.65,  # enter-open threshold
    T_low: float = 0.45,  # exit-open threshold
    min_run: int = 2,  # minimum samples to keep a new state (dwell)
    gap_reset_hours: float = 96.0,  # reset state across large gaps
) -> np.ndarray:
    """
    Returns a 0/1 numpy array of states (0=closed, 1=open) using hysteresis.
    - Start state chosen by first non-NaN prob with T_high/T_low logic.
    - If a flip occurs, it must persist for `min_run` samples or it is reverted.
    - Across gaps > gap_reset_hours, state is re-initialized (no carry-over).
    """
    assert len(times) == len(probs)
    n = len(probs)
    states = np.zeros(n, dtype=np.int8)

    # Precompute gaps
    t = pd.to_datetime(times).reset_index(drop=True)
    dts = t.diff().dt.total_seconds().fillna(0).to_numpy()
    gap_thresh = gap_reset_hours * 3600.0

    # Helper: pick initial state from a probability value
    def init_from_p(p):
        if np.isnan(p):  # unknown -> default closed
            return 0
        if p >= T_high:  # confidently open
            return 1
        if p <= T_low:  # confidently closed
            return 0
        # undecided region: default closed (tunable)
        return 0

    # Initialize first segment
    states[0] = init_from_p(probs[0])
    cur = states[0]
    run_len = 1  # current run length in `cur` state

    for i in range(1, n):
        # Reset across large gaps
        if dts[i] > gap_thresh:
            cur = init_from_p(probs[i])
            states[i] = cur
            run_len = 1
            continue

        p = probs[i]
        prev = cur

        # Hysteresis rule
        if cur == 1:
            # currently open; need strong evidence to close
            cur = 0 if (not np.isnan(p) and p <= T_low) else 1
        else:
            # currently closed; need strong evidence to open
            cur = 1 if (not np.isnan(p) and p >= T_high) else 0

        if cur == prev:
            run_len += 1
            states[i] = cur
        else:
            # Tentative flip: enforce minimum duration
            # Look ahead up to min_run samples (bounded by array end and gaps)
            end = i + min_run
            # Stop early if a large gap occurs in the lookahead window
            j = i
            ok = True
            while j < min(end, n):
                if j > i and dts[j] > gap_thresh:
                    ok = False  # do not enforce across a gap; cancel flip
                    break
                pj = probs[j]
                # Must keep satisfying the hysteresis condition in the new state
                if cur == 1 and (np.isnan(pj) or pj < T_high):
                    ok = False
                    break
                if cur == 0 and (np.isnan(pj) or pj > T_low):
                    ok = False
                    break
                j += 1

            if ok and j <= n:
                # Commit flip: fill the run [i, j-1] with new state
                states[i:j] = cur
                run_len = j - i
                # Continue from j-1 as last written; loop moves to j next
                # (we won't skip indices; we just set current state here)
            else:
                # Cancel flip, keep previous state
                cur = prev
                states[i] = cur
                run_len += 1

    return states


def extract_changes(
    times: pd.Series,
    states: np.ndarray,
    gap_reset_hours: float = 96.0,
) -> list[tuple[pd.Timestamp, int]]:
    """Return change timestamps with new-state labels (0 or 1).
    Resets across large gaps. Returns a list of
    (timestamp, new_state) tuples so callers can distinguish 0→1 vs 1→0.
    """
    t = pd.to_datetime(times).reset_index(drop=True)
    dts = t.diff().dt.total_seconds().fillna(0)
    gap_s = gap_reset_hours * 3600.0
    changes: list[tuple[pd.Timestamp, int]] = []

    prev_state = int(states[0])
    prev_time = t.iat[0]
    for i in range(1, len(states)):
        curr_time = t.iat[i]
        s = int(states[i])
        if dts.iat[i] > gap_s:
            # Large gap: if state differs after the gap, assume the change happened
            # at the midpoint between the last and current timestamps.
            if s != prev_state:
                mid = prev_time + (curr_time - prev_time) / 2
                changes.append((mid, s))
            prev_state = s
            prev_time = curr_time
            continue

        # Normal (non-gap) transition detection at the boundary sample
        if s != prev_state:
            changes.append((curr_time, s))
            prev_state = s
            prev_time = curr_time

    return changes


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matched_pairs: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]]  # (gt, pred, diff)
    mae_days: float | None


def match_events_with_tolerance(
    gt_events: list[tuple[pd.Timestamp, int]],
    pred_events: list[tuple[pd.Timestamp, int]],
    tol_days: float = 3.0,
) -> MatchResult:
    """Greedy nearest-neighbor matching within ±tol_days (one-to-one).
    Events are (time, new_state) tuples.
    Only match events with the same new_state (i.e., 0→1 vs 1→0 distinction).
    """
    tol = pd.Timedelta(days=tol_days)

    # Normalize to (time, state_or_None)
    def _norm(ev_list):
        out: list[tuple[pd.Timestamp, int | None]] = []
        for e in ev_list:
            if isinstance(e, tuple):
                out.append((pd.to_datetime(e[0]), int(e[1])))
            else:
                out.append((pd.to_datetime(e), None))
        return out

    gt_norm = sorted(_norm(gt_events), key=lambda x: x[0])
    pr_norm = sorted(_norm(pred_events), key=lambda x: x[0])

    used_pred: set[int] = set()
    pairs: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]] = []

    for i, (gt_t, gt_s) in enumerate(gt_norm):
        best = None
        best_abs = None
        for j, (pr_t, pr_s) in enumerate(pr_norm):
            if j in used_pred:
                continue
            if (gt_s is not None) and (pr_s is not None) and (gt_s != pr_s):
                continue
            diff = pr_t - gt_t
            if abs(diff) <= tol:
                ad = abs(diff)
                if (best_abs is None) or (ad < best_abs):
                    best = (j, pr_t, diff)
                    best_abs = ad
        if best is not None:
            j, pr_t, diff = best
            used_pred.add(j)
            pairs.append((gt_t, pr_t, diff))

    tp = len(pairs)
    fn = len(gt_norm) - tp
    fp = len(pr_norm) - tp
    mae_days = (
        float(np.mean([abs(d).days + abs(d).seconds / 86400 for _, _, d in pairs]))
        if pairs
        else None
    )
    return MatchResult(tp=tp, fp=fp, fn=fn, matched_pairs=pairs, mae_days=mae_days)


def evaluate_changes_by_site(
    df: pd.DataFrame,
    site_col: str = "region",
    time_col: str = "acquired",
    prob_col: str = "y_prob",
    ytrue_col: str = "y_true",
    T_high: float = 0.7,
    T_low: float = 0.3,
    min_run: int = 2,
    gap_reset_hours: float = 96.0,
    tol_days: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (per_site_df, summary_dict).
    df must contain [site_col, time_col, prob_col, ytrue_col], sorted by time per site.
    """
    records = []
    for site, g in df.groupby(site_col):
        g = g.sort_values(time_col)
        times = g[time_col]
        probs = g[prob_col].to_numpy(dtype=float)
        y_true = g[ytrue_col].to_numpy(dtype=int)

        # smooth predictions
        states_pred = hysteresis_decode(times, probs, T_high, T_low, min_run, gap_reset_hours)

        # (optional) smooth labels as well, or keep raw? — you asked to smooth per-frame labels too:
        states_true = hysteresis_decode(
            times, y_true.astype(float), T_high, T_low, min_run, gap_reset_hours
        )

        # extract change timestamps
        gt_events = extract_changes(times, states_true, gap_reset_hours)
        pred_events = extract_changes(times, states_pred, gap_reset_hours)

        res = match_events_with_tolerance(gt_events, pred_events, tol_days=tol_days)

        prec = res.tp / (res.tp + res.fp) if (res.tp + res.fp) else 0.0
        rec = res.tp / (res.tp + res.fn) if (res.tp + res.fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        records.append(
            {
                "site": site,
                "gt_events": len(gt_events),
                "pred_events": len(pred_events),
                "tp": res.tp,
                "fp": res.fp,
                "fn": res.fn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "mae_days": res.mae_days,
            }
        )

    per_site = pd.DataFrame.from_records(records).sort_values("site")

    # macro and micro summaries
    macro = {
        "precision_macro": per_site["precision"].mean(),
        "recall_macro": per_site["recall"].mean(),
        "f1_macro": per_site["f1"].mean(),
        "mae_days_macro": per_site["mae_days"].mean(skipna=True),
    }
    tp = per_site["tp"].sum()
    fp = per_site["fp"].sum()
    fn = per_site["fn"].sum()
    micro = {
        "precision_micro": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall_micro": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1_micro": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    }
    return per_site, {"macro": macro, "micro": micro}


# for region in tdf.region.unique().tolist():
#     print(region)

#     rdf = tdf[tdf.region == region].copy()
#     smooth_labels = hysteresis_decode(
#         rdf.acquired,
#         rdf.y_true.to_numpy(),
#         T_high=0.8,
#         T_low=0.2,
#     )
#     time_preds = hysteresis_decode(
#         rdf.acquired,
#         rdf.y_prob.to_numpy(),
#         T_high=0.8,
#         T_low=0.2,
#     )

#     print(rdf.correct.mean().round(3))
#     print((time_preds == smooth_labels).mean().round(3))

#     print("num events (no smooth)", len(extract_changes(rdf.acquired, rdf.y_true.to_numpy())))
#     result = match_events_with_tolerance(
#         extract_changes(rdf.acquired, rdf.y_true.to_numpy()),
#         extract_changes(rdf.acquired, rdf.y_pred.to_numpy()),
#     )

#     print(result.tp, result.fp)

#     print("num events (smooth)", len(extract_changes(rdf.acquired, smooth_labels)))
#     result = match_events_with_tolerance(
#         extract_changes(rdf.acquired, smooth_labels),
#         extract_changes(rdf.acquired, time_preds),
#     )

#     print(result.tp, result.fp)
#     print("")


tdf = pd.read_csv(
    "/Users/kyledorman/data/results/estuary/train/20251008-151833/timeseries_preds.csv"
)
tdf["acquired"] = tdf.source_tif.apply(lambda a: parse_dt_from_pth(Path(a)))
tdf = tdf.sort_values("acquired").reset_index(drop=True)

rr = evaluate_changes_by_site(tdf)

print(rr)
