# %% [markdown]
# # Initial Opening Flow Threshold Exploration
#
# This notebook-style script assumes these objects already exist in your
# Jupyter kernel:
#
# - `events_df`
# - `sim_all`
# - `trace_closed`
# - `region_levels_closed`
#
# Goal:
#
# 1. Find the first extended `closed_to_open` event per region/site per year.
# 2. Join that event to the driver/model row used by `pymc_all_closed`.
# 3. Invert the closed-state PyMC model to estimate the model-scale
#    `flow_window` threshold needed to reach a target opening probability.
# 4. Map that model-scale/ranked threshold back to an empirical raw flow value.
#
# Main assumption to check: by default, an "extended opening" is a
# `closed_to_open` row where `opening_eval_window` is truthy. If that flag means
# something else in your event table, change `USE_OPENING_EVAL_WINDOW` or set
# `MIN_FOLLOWING_OPEN_DAYS`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Event selection.
OPENING_TRANSITION = "closed_to_open"
EVENT_ANCHOR_DATE_COL = "window_end_date"
USE_OPENING_EVAL_WINDOW = True
MIN_FOLLOWING_OPEN_DAYS = 14  # e.g. 14, 30, or None to only use the flag above

# Model / driver configuration.
MODEL_NAME = "pymc_all_closed"
MODEL_FLOW_COL = "flow_window"
RAW_FLOW_COL = "flow_sum_rolling_7d"
FIXED_COLS = ["wave_window", "month_sin", "month_cos"]
TARGET_OPEN_PROB = 0.70

# Raw-flow inversion.
# Keep this as ("region",) first. If you want year-specific empirical inversions,
# try ("region", "year") after checking sample sizes.
RAW_LOOKUP_GROUP_COLS = ("region",)  # "water_year"

pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 180)


# %%
def expit(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.asarray(p)
    return np.log(p / (1 - p))


def require_columns(df, required, df_name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def truthy_series(s):
    """Handle booleans, 0/1, and string-ish booleans."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0)

    normalized = s.astype("string").str.strip().str.lower()
    return normalized.isin(["true", "t", "yes", "y", "1"])


def prepare_events(events_df, anchor_date_col=EVENT_ANCHOR_DATE_COL):
    required = [
        "region",
        "window_start_date",
        "window_end_date",
        "window_start_state",
        "window_end_state",
        "transition",
    ]
    require_columns(events_df, required, "events_df")

    events = events_df.copy()
    events["window_start_date"] = pd.to_datetime(events["window_start_date"])
    events["window_end_date"] = pd.to_datetime(events["window_end_date"])
    events["event_date"] = pd.to_datetime(events[anchor_date_col])
    events["event_water_year"] = events["event_date"].dt.year + events["event_date"].dt.month.ge(
        10
    ).astype(int)
    events["window_days"] = (events["window_end_date"] - events["window_start_date"]).dt.days

    return events.sort_values(["region", "window_start_date", "window_end_date"]).reset_index(
        drop=True
    )


def add_following_open_spell(events):
    """For each closed_to_open event, estimate how long the next open spell lasts."""
    out = events.copy()
    out["following_open_until_date"] = pd.NaT
    out["following_open_days"] = np.nan
    out["following_open_is_censored"] = False

    for _, g in out.groupby("region", sort=False):
        g = g.sort_values(["window_start_date", "window_end_date"])
        region_end = g["window_end_date"].max()
        closing_events = g[g["transition"].eq("open_to_closed")]

        for idx, row in g[g["transition"].eq(OPENING_TRANSITION)].iterrows():
            opening_date = row["window_end_date"]
            later_closings = closing_events[closing_events["window_start_date"].ge(opening_date)]

            if later_closings.empty:
                open_until = region_end
                censored = True
            else:
                open_until = later_closings.iloc[0]["window_end_date"]
                censored = False

            out.loc[idx, "following_open_until_date"] = open_until
            out.loc[idx, "following_open_days"] = (open_until - opening_date).days
            out.loc[idx, "following_open_is_censored"] = censored

    return out


def select_first_extended_openings(events):
    mask = events["transition"].eq(OPENING_TRANSITION)

    if USE_OPENING_EVAL_WINDOW and "opening_eval_window" in events.columns:
        mask &= truthy_series(events["opening_eval_window"])

    if MIN_FOLLOWING_OPEN_DAYS is not None:
        mask &= events["following_open_days"].ge(MIN_FOLLOWING_OPEN_DAYS)

    openings = events.loc[mask].copy()

    first_openings = (
        openings.sort_values(["region", "event_water_year", "event_date"])
        .groupby(["region", "event_water_year"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )

    return openings, first_openings


# %%
base = Path("/Users/kyledorman/data/results/estuary/train/20260305-095554/seasonal_drivers")
events_df = pd.read_parquet(base / "transition_event_windows.parquet")
events = prepare_events(events_df)
events = add_following_open_spell(events)
opening_events, first_openings = select_first_extended_openings(events)

print("All event transitions:")
display(events["transition"].value_counts(dropna=False))

print("Candidate extended opening events:")
display(opening_events["transition"].value_counts(dropna=False))

print("First extended opening per region/year:")
display(
    first_openings[
        [
            "region",
            "event_water_year",
            "event_date",
            "window_start_date",
            "window_end_date",
            "obs_gap_days",
            "window_start_state",
            "window_end_state",
            "window_start_p_open",
            "window_end_p_open",
            "opening_eval_window",
            "opening_change",
            "following_open_days",
            "following_open_is_censored",
        ]
    ].head(20)
)

print(
    {
        "regions_with_first_opening": first_openings["region"].nunique(),
        "region_year_first_openings": len(first_openings),
        "years": sorted(first_openings["event_water_year"].dropna().unique().tolist()),
    }
)


# %%
def prepare_sim(sim_all):
    required = ["region", "date", MODEL_FLOW_COL, RAW_FLOW_COL, *FIXED_COLS]
    require_columns(sim_all, required, "sim_all")

    sim = sim_all.copy()
    sim["date"] = pd.to_datetime(sim["date"])

    if "model" in sim.columns and MODEL_NAME in set(sim["model"].dropna()):
        sim = sim[sim["model"].eq(MODEL_NAME)].copy()

    if "year" not in sim.columns:
        sim["year"] = sim["date"].dt.year

    return sim


def attach_driver_rows(first_openings, sim):
    driver_cols = [
        "region",
        "date",
        MODEL_FLOW_COL,
        RAW_FLOW_COL,
        *FIXED_COLS,
        "year",
        "water_year",
        "p_if_closed",
        "p_if_open",
        "p_open_sim",
        "prev_state",
        "prev_p_open",
        "confident_state_lag7",
    ]
    driver_cols = [col for col in driver_cols if col in sim.columns]

    joined = first_openings.merge(
        sim[driver_cols],
        left_on=["region", "event_date"],
        right_on=["region", "date"],
        how="left",
        validate="m:1",
    )

    joined["missing_driver_row"] = joined["date"].isna()
    return joined


sim = prepare_sim(sim_all)
first_opening_drivers = attach_driver_rows(first_openings, sim)

print("Driver rows joined to first openings:")
display(
    first_opening_drivers[
        [
            "region",
            "event_water_year",
            "event_date",
            "missing_driver_row",
            MODEL_FLOW_COL,
            RAW_FLOW_COL,
            *FIXED_COLS,
            *[
                col
                for col in ["p_if_closed", "prev_state", "confident_state_lag7"]
                if col in first_opening_drivers.columns
            ],
        ]
    ].head(20)
)

print("Missing exact driver joins:")
display(
    first_opening_drivers.loc[
        first_opening_drivers["missing_driver_row"],
        ["region", "event_water_year", "event_date"],
    ].head(30)
)


# %%
def add_nearest_driver_rows_for_missing(first_opening_drivers, sim, tolerance_days=3):
    """Optional fallback if event dates and driver dates do not line up exactly."""
    missing = first_opening_drivers[first_opening_drivers["missing_driver_row"]].copy()
    matched = first_opening_drivers[~first_opening_drivers["missing_driver_row"]].copy()

    if missing.empty:
        return first_opening_drivers

    driver_cols = [
        "region",
        "date",
        MODEL_FLOW_COL,
        RAW_FLOW_COL,
        *FIXED_COLS,
        "year",
        "water_year",
        "p_if_closed",
        "p_if_open",
        "p_open_sim",
        "prev_state",
        "prev_p_open",
        "confident_state_lag7",
    ]
    driver_cols = [col for col in driver_cols if col in sim.columns]

    nearest = pd.merge_asof(
        missing.drop(
            columns=[col for col in driver_cols if col in missing.columns], errors="ignore"
        ).sort_values(["region", "event_date"]),
        sim[driver_cols].sort_values(["region", "date"]),
        left_on="event_date",
        right_on="date",
        by="region",
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    nearest["missing_driver_row"] = nearest["date"].isna()
    nearest["driver_join_method"] = f"nearest_{tolerance_days}d"

    matched["driver_join_method"] = "exact"

    return (
        pd.concat([matched, nearest], ignore_index=True)
        .sort_values(["region", "event_water_year", "event_date"])
        .reset_index(drop=True)
    )


# Uncomment if exact event-date joins miss driver rows you need.
# first_opening_drivers = add_nearest_driver_rows_for_missing(
#     first_opening_drivers,
#     sim,
#     tolerance_days=3,
# )


# %%
def extract_posterior_mean_params(trace_closed):
    posterior = trace_closed.posterior

    params = {
        "alpha": posterior["alpha"].mean(dim=("chain", "draw")).values,
        "beta_fixed": posterior["beta_fixed"].mean(dim=("chain", "draw")).values,
        "beta_flow_window": posterior["beta_flow_window"].mean(dim=("chain", "draw")).values,
        "region_intercept": posterior["region_intercept"].mean(dim=("chain", "draw")).values,
        "region_flow_window_dev": posterior["region_flow_window_dev"]
        .mean(dim=("chain", "draw"))
        .values,
    }

    params["alpha"] = float(np.asarray(params["alpha"]).squeeze())
    params["beta_fixed"] = np.asarray(params["beta_fixed"], dtype=float).reshape(-1)
    params["beta_flow_window"] = float(np.asarray(params["beta_flow_window"]).squeeze())
    params["region_intercept"] = np.asarray(params["region_intercept"], dtype=float).reshape(-1)
    params["region_flow_window_dev"] = np.asarray(
        params["region_flow_window_dev"], dtype=float
    ).reshape(-1)

    return params


def add_model_scale_flow_thresholds(
    df,
    trace_closed,
    region_levels_closed,
    fixed_cols=FIXED_COLS,
    flow_col=MODEL_FLOW_COL,
    target_open_prob=TARGET_OPEN_PROB,
):
    require_columns(df, ["region", flow_col, *fixed_cols], "event-driver dataframe")

    out = df.copy()
    params = extract_posterior_mean_params(trace_closed)
    region_lookup = pd.Series(
        np.arange(len(region_levels_closed)),
        index=pd.Index(region_levels_closed),
    )

    out["region_idx"] = out["region"].map(region_lookup)
    valid = out["region_idx"].notna() & out[[flow_col, *fixed_cols]].notna().all(axis=1)

    out["closed_model_base_eta_no_flow"] = np.nan
    out["flow_window_slope"] = np.nan
    out["observed_p_open_from_closed_model"] = np.nan
    out["flow_window_threshold_model_scale"] = np.nan
    out["flow_threshold_status"] = "missing_inputs"

    idx = out.index[valid]
    if len(idx) == 0:
        return out

    region_idx = out.loc[idx, "region_idx"].astype(int).to_numpy()
    X_fixed = out.loc[idx, fixed_cols].to_numpy(dtype=float)
    observed_flow = out.loc[idx, flow_col].to_numpy(dtype=float)

    base_eta = (
        params["alpha"] + params["region_intercept"][region_idx] + X_fixed @ params["beta_fixed"]
    )
    flow_slope = params["beta_flow_window"] + params["region_flow_window_dev"][region_idx]
    eta_observed = base_eta + observed_flow * flow_slope

    threshold = (logit(target_open_prob) - base_eta) / flow_slope

    out.loc[idx, "closed_model_base_eta_no_flow"] = base_eta
    out.loc[idx, "flow_window_slope"] = flow_slope
    out.loc[idx, "observed_p_open_from_closed_model"] = expit(eta_observed)
    out.loc[idx, "flow_window_threshold_model_scale"] = threshold
    out.loc[idx, "flow_threshold_status"] = np.where(
        flow_slope > 0,
        "minimum_flow_reaches_target",
        "negative_or_zero_flow_slope_check_model",
    )
    out.loc[idx, "observed_minus_threshold_model_scale"] = observed_flow - threshold
    out.loc[idx, "observed_meets_model_threshold"] = observed_flow >= threshold

    return out


threshold_results = add_model_scale_flow_thresholds(
    first_opening_drivers,
    trace_closed=trace_closed,
    region_levels_closed=region_levels_closed,
)

display(
    threshold_results[
        [
            "region",
            "event_water_year",
            "event_date",
            MODEL_FLOW_COL,
            "flow_window_threshold_model_scale",
            "observed_minus_threshold_model_scale",
            "observed_p_open_from_closed_model",
            "flow_window_slope",
            "flow_threshold_status",
        ]
    ].head(30)
)


# %%
def make_empirical_flow_lookup(sim, group_cols=RAW_LOOKUP_GROUP_COLS):
    require_columns(sim, [*group_cols, MODEL_FLOW_COL, RAW_FLOW_COL], "sim lookup")

    lookup = sim[[*group_cols, MODEL_FLOW_COL, RAW_FLOW_COL]].dropna().copy()

    # If multiple raw values land on the same model-scale flow value, use the
    # median raw flow at that model-scale value before interpolation.
    lookup = (
        lookup.groupby([*group_cols, MODEL_FLOW_COL], as_index=False)[RAW_FLOW_COL]
        .median()
        .sort_values([*group_cols, MODEL_FLOW_COL])
    )

    return lookup


def estimate_raw_flow_for_thresholds(
    threshold_df,
    sim,
    group_cols=RAW_LOOKUP_GROUP_COLS,
    threshold_col="flow_window_threshold_model_scale",
):
    lookup = make_empirical_flow_lookup(sim, group_cols=group_cols)

    out = threshold_df.copy()
    out["flow_threshold_raw_estimate"] = np.nan
    out["flow_threshold_raw_status"] = "missing_threshold_or_lookup"
    out["flow_lookup_model_scale_min"] = np.nan
    out["flow_lookup_model_scale_max"] = np.nan

    lookup_groups = {
        key if isinstance(key, tuple) else (key,): g
        for key, g in lookup.groupby(list(group_cols), sort=False)
    }

    for idx, row in out.iterrows():
        threshold = row[threshold_col]
        if pd.isna(threshold):
            continue

        key = tuple(row[col] for col in group_cols)
        g = lookup_groups.get(key)
        if g is None or g.empty:
            continue

        x = g[MODEL_FLOW_COL].to_numpy(dtype=float)
        y = g[RAW_FLOW_COL].to_numpy(dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        out.loc[idx, "flow_lookup_model_scale_min"] = x_min
        out.loc[idx, "flow_lookup_model_scale_max"] = x_max

        if threshold < x_min:
            clipped_threshold = x_min
            status = "below_observed_model_scale_range"
        elif threshold > x_max:
            clipped_threshold = x_max
            status = "above_observed_model_scale_range"
        else:
            clipped_threshold = threshold
            status = "interpolated_within_observed_range"

        out.loc[idx, "flow_threshold_raw_estimate"] = np.interp(
            clipped_threshold,
            x,
            y,
        )
        out.loc[idx, "flow_threshold_raw_status"] = status

    return out


threshold_results = estimate_raw_flow_for_thresholds(
    threshold_results,
    sim=sim,
    group_cols=RAW_LOOKUP_GROUP_COLS,
)

display(
    threshold_results[
        [
            "region",
            "event_water_year",
            "event_date",
            MODEL_FLOW_COL,
            RAW_FLOW_COL,
            "flow_window_threshold_model_scale",
            "flow_threshold_raw_estimate",
            "flow_threshold_raw_status",
            "observed_p_open_from_closed_model",
            "observed_meets_model_threshold",
        ]
    ].head(30)
)


# %% [markdown]
# ## Diagnostics
#
# These summaries are the first things to check:
#
# - `negative_or_zero_flow_slope_check_model`: the model does not imply that
#   more `flow_window` increases opening probability for that region.
# - `above_observed_model_scale_range`: the estimated threshold is higher than
#   any observed `flow_window` in the lookup group, so the raw value is clipped
#   to the maximum observed raw-flow estimate.
# - `below_observed_model_scale_range`: the target probability is reached below
#   the observed flow range.
# - `missing_driver_row`: the event date did not exactly join to `sim_all`.

# %%
summary_cols = [
    "flow_window_threshold_model_scale",
    "flow_threshold_raw_estimate",
    "observed_p_open_from_closed_model",
    "observed_minus_threshold_model_scale",
]

display(
    threshold_results.groupby("event_water_year")[summary_cols]
    .agg(["count", "median", "min", "max"])
    .round(3)
)

display(threshold_results["flow_threshold_status"].value_counts(dropna=False))
display(threshold_results["flow_threshold_raw_status"].value_counts(dropna=False))
display(threshold_results["observed_meets_model_threshold"].value_counts(dropna=False))

problem_rows = threshold_results[
    threshold_results["missing_driver_row"]
    | threshold_results["flow_threshold_status"].ne("minimum_flow_reaches_target")
    | threshold_results["flow_threshold_raw_status"].ne("interpolated_within_observed_range")
]

display(
    problem_rows[
        [
            "region",
            "event_water_year",
            "event_date",
            "missing_driver_row",
            "flow_threshold_status",
            "flow_threshold_raw_status",
            MODEL_FLOW_COL,
            "flow_window_threshold_model_scale",
            RAW_FLOW_COL,
            "flow_threshold_raw_estimate",
        ]
    ].head(50)
)


# %%
plot_df = threshold_results.dropna(
    subset=[MODEL_FLOW_COL, RAW_FLOW_COL, "flow_window_threshold_model_scale"]
).copy()

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    plot_df[MODEL_FLOW_COL],
    plot_df[RAW_FLOW_COL],
    s=35,
    alpha=0.75,
    label="Observed event-day flow",
)
ax.scatter(
    plot_df["flow_window_threshold_model_scale"],
    plot_df["flow_threshold_raw_estimate"],
    s=55,
    marker="x",
    label=f"Model threshold at p={TARGET_OPEN_PROB:.2f}",
)
ax.set_xlabel(f"{MODEL_FLOW_COL} model-scale/ranked value")
ax.set_ylabel(RAW_FLOW_COL)
ax.set_title("Initial opening flow: observed vs model-implied threshold")
ax.legend()
plt.show()


# %%
top_regions = threshold_results["region"].value_counts().head(12).index

facet_df = threshold_results[threshold_results["region"].isin(top_regions)].copy()

if not facet_df.empty:
    ncols = 3
    nrows = int(np.ceil(len(top_regions) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 3.3 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, region in zip(axes, top_regions):
        g = facet_df[facet_df["region"].eq(region)]
        ax.scatter(g["event_water_year"], g[RAW_FLOW_COL], label="observed raw", alpha=0.75)
        ax.scatter(
            g["event_water_year"],
            g["flow_threshold_raw_estimate"],
            label="threshold raw",
            marker="x",
        )
        ax.set_title(region)
        ax.set_xlabel("year")
        ax.set_ylabel(RAW_FLOW_COL)

    for ax in axes[len(top_regions) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# %% [markdown]
# ## Optional: posterior uncertainty for the flow threshold
#
# The core cells above mirror your prediction function by using posterior means.
# This optional cell samples posterior draws and computes a per-event threshold
# interval. It is slower but gives a better sense of uncertainty.


# %%
def _posterior_samples(var, sample_idx):
    arr = var.stack(sample=("chain", "draw")).isel(sample=sample_idx)
    other_dims = [dim for dim in arr.dims if dim != "sample"]
    arr = arr.transpose("sample", *other_dims)
    return np.asarray(arr.values)


def add_posterior_threshold_intervals(
    df,
    trace_closed,
    region_levels_closed,
    fixed_cols=FIXED_COLS,
    target_open_prob=TARGET_OPEN_PROB,
    max_draws=1000,
    random_seed=42,
):
    out = df.copy()
    posterior = trace_closed.posterior
    n_samples = posterior.sizes["chain"] * posterior.sizes["draw"]
    rng = np.random.default_rng(random_seed)
    sample_idx = np.sort(rng.choice(n_samples, size=min(max_draws, n_samples), replace=False))

    alpha = _posterior_samples(posterior["alpha"], sample_idx).reshape(len(sample_idx))
    beta_fixed = _posterior_samples(posterior["beta_fixed"], sample_idx)
    beta_flow = _posterior_samples(posterior["beta_flow_window"], sample_idx).reshape(
        len(sample_idx)
    )
    region_intercept = _posterior_samples(posterior["region_intercept"], sample_idx)
    region_flow_dev = _posterior_samples(posterior["region_flow_window_dev"], sample_idx)

    region_lookup = pd.Series(
        np.arange(len(region_levels_closed)),
        index=pd.Index(region_levels_closed),
    )

    out["threshold_model_scale_p05"] = np.nan
    out["threshold_model_scale_p50"] = np.nan
    out["threshold_model_scale_p95"] = np.nan
    out["threshold_positive_slope_share"] = np.nan

    valid = out["region"].map(region_lookup).notna() & out[fixed_cols].notna().all(axis=1)

    for idx, row in out[valid].iterrows():
        region_idx = int(region_lookup.loc[row["region"]])
        x_fixed = row[fixed_cols].to_numpy(dtype=float)

        base_eta = (
            alpha + region_intercept[:, region_idx] + np.einsum("sf,f->s", beta_fixed, x_fixed)
        )
        slope = beta_flow + region_flow_dev[:, region_idx]
        threshold = (logit(target_open_prob) - base_eta) / slope
        threshold = threshold[np.isfinite(threshold)]

        if len(threshold) == 0:
            continue

        out.loc[idx, "threshold_model_scale_p05"] = np.quantile(threshold, 0.05)
        out.loc[idx, "threshold_model_scale_p50"] = np.quantile(threshold, 0.50)
        out.loc[idx, "threshold_model_scale_p95"] = np.quantile(threshold, 0.95)
        out.loc[idx, "threshold_positive_slope_share"] = np.mean(slope > 0)

    return out


# Uncomment when you want uncertainty intervals.
# threshold_results_with_uncertainty = add_posterior_threshold_intervals(
#     threshold_results,
#     trace_closed=trace_closed,
#     region_levels_closed=region_levels_closed,
#     max_draws=1000,
# )
# display(
#     threshold_results_with_uncertainty[
#         [
#             "region",
#             "event_water_year",
#             "event_date",
#             "flow_window_threshold_model_scale",
#             "threshold_model_scale_p05",
#             "threshold_model_scale_p50",
#             "threshold_model_scale_p95",
#             "threshold_positive_slope_share",
#         ]
#     ].head(30)
# )


# %% [markdown]
# ## Optional export

# %%
# threshold_results.to_parquet("initial_opening_flow_thresholds.parquet", index=False)
# threshold_results.to_csv("initial_opening_flow_thresholds.csv", index=False)
