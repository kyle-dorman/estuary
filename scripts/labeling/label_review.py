#!/usr/bin/env python3
"""
Streamlit reviewer for estuary mouth-state predictions.

Run:
  streamlit run review_predictions_app.py -- --predictions /path/to/preds.csv

CSV columns expected (at minimum):
  - source_tif (path to .tif)
  - acquired (timestamp)
  - region
  - year
  - y_pred
  - asset_id
  - (optional) y_true
  - probability columns like p_closed, p_perched_open, p_open, etc.

Image loading:
  uses estuary.util.img.tif_to_rgb(tif_path) -> np.ndarray (H,W,3) or (H,W,4)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st

from estuary.util.constants import FALSE_COLOR_4, FALSE_COLOR_8


# -----------------------------
# CLI parsing (Streamlit style)
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--predictions", type=str, default="", help="Path to predictions CSV")
    return parser.parse_known_args()[0]


# -----------------------------
# Helpers
# -----------------------------
def _coerce_dt_utc(x: pd.Series) -> pd.Series:
    # Accept strings, timestamps, etc. Coerce to tz-aware UTC.
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    return dt


def _find_prob_cols(df: pd.DataFrame) -> list[str]:
    # Any column starting with "p_" is treated as a probability column.
    return [c for c in df.columns if c.startswith("p_")]


def tif_to_rgb(
    pth: Path,
    p_low: tuple[int, ...],
    p_high: tuple[int, ...],
) -> np.ndarray:
    with rasterio.open(pth) as src:
        data = src.read(out_dtype=np.float32)
        nodata = src.read_masks(1) == 0
        if nodata.all():
            return np.zeros((*nodata.shape, 3), dtype=np.uint8)

        return false_color(data, nodata, p_low=p_low, p_high=p_high)


def false_color(
    data: np.ndarray,
    nodata: np.ndarray,
    p_low: tuple[int, ...],
    p_high: tuple[int, ...],
):
    channels = FALSE_COLOR_4 if len(data) == 4 else FALSE_COLOR_8
    img = normalized_image_3_channel(data, nodata, channels, p_low=p_low, p_high=p_high).transpose(
        (1, 2, 0)
    )

    k = 1.5
    img = np.tanh(k * img) / np.tanh(k)

    img = np.array(img * 255, dtype=np.uint8)

    return img


def normalized_image_3_channel(
    data: np.ndarray,
    nodata: np.ndarray,
    bands: tuple[int, int, int],
    p_low: tuple[int, ...],
    p_high: tuple[int, ...],
) -> np.ndarray:
    img = np.array([data[b] for b in bands])
    img = masked_contrast_stretch(img, ~nodata, p_low=p_low, p_high=p_high)

    for i in range(3):
        img[i][nodata] = 0

    return img


def masked_contrast_stretch(
    image: np.ndarray, mask: np.ndarray, p_low: tuple[int, ...], p_high: tuple[int, ...]
) -> np.ndarray:
    """Perform contrast stretching using percentiles."""
    image = image.astype(np.float32)
    orig_shape = image.shape
    if len(orig_shape) == 2:
        image = image[None]
    for idx in range(image.shape[0]):
        channel = image[idx]
        v_min, v_max = np.percentile(channel[mask], (p_low[idx], p_high[idx]))

        # If no p_low use 0 (absolute)
        if p_low is None:
            v_min = 0

        image[idx] = np.clip((channel - v_min) / (v_max - v_min), 0, 1)

    if len(orig_shape) == 2:
        image = image[0]

    return image


@st.cache_data(show_spinner=False)
def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize common columns
    if "acquired" in df.columns:
        df["acquired"] = _coerce_dt_utc(df["acquired"])
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "region" in df.columns:
        # keep as string for consistent filtering
        df["region"] = df["region"].astype(str)
    # y_pred / y_true can be int or string; keep as string for UI multiselect
    if "y_pred" in df.columns:
        df["y_pred"] = df["y_pred"].astype(str)
    if "y_true" in df.columns:
        df["y_true"] = df["y_true"].astype(str)

    # Ensure source_tif exists
    if "source_tif" not in df.columns:
        raise ValueError("CSV must contain a 'source_tif' column (path to tif).")

    return df


@st.cache_data(show_spinner=False)
def load_rgb_image(
    tif_path: str,
    p_low: tuple[int, ...],
    p_high: tuple[int, ...],
) -> np.ndarray:
    return tif_to_rgb(Path(tif_path), p_low, p_high)


def make_filtered_df(
    df: pd.DataFrame,
    *,
    y_true_vals: list[str] | None,
    y_pred_vals: list[str] | None,
    regions: list[str] | None,
    years: list[int] | None,
) -> pd.DataFrame:
    out = df

    if y_true_vals is not None and len(y_true_vals) > 0 and "y_true" in out.columns:
        out = out[out["y_true"].isin(y_true_vals)]
    if y_pred_vals is not None and len(y_pred_vals) > 0 and "y_pred" in out.columns:
        out = out[out["y_pred"].isin(y_pred_vals)]
    if regions is not None and len(regions) > 0 and "region" in out.columns:
        out = out[out["region"].isin(regions)]
    if years is not None and len(years) > 0 and "year" in out.columns:
        out = out[out["year"].isin(pd.Series(years, dtype="Int64"))]

    return out


def build_sequence(
    df: pd.DataFrame,
    *,
    mode: str,
    seed: int,
) -> pd.DataFrame:
    if df.empty:
        return df

    # Sort by timestamp
    if mode == "date":
        if "acquired" in df.columns:
            return df.sort_values("acquired", kind="mergesort")
        return df.sort_index(kind="mergesort")

    # Sort by region (and then by acquired if available)
    if mode == "region":
        if "region" in df.columns:
            if "acquired" in df.columns:
                return df.sort_values(["region", "acquired"], kind="mergesort")
            return df.sort_values("region", kind="mergesort")
        return df

    # Sort by a specific column, descending by default (used for p_* columns)
    if mode.startswith("col:"):
        col = mode.split(":", 1)[1]
        if col in df.columns:
            # Put NaNs at the end for easier review
            return df.sort_values(col, ascending=False, na_position="last", kind="mergesort")
        # fallback
        return df

    # random
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    return df.iloc[order]


def init_state():
    st.session_state.setdefault("seq_key", None)
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("jump_idx", 0)


def reset_sequence(seq_key: str):
    st.session_state["seq_key"] = seq_key
    st.session_state["idx"] = 0
    st.session_state["jump_idx"] = 0


# -----------------------------
# App
# -----------------------------
def main():
    args = parse_args()
    st.set_page_config(layout="wide")

    init_state()

    with st.sidebar:
        st.header("Data")
        csv_path = st.text_input("Predictions CSV path", value=args.predictions)
        st.caption("Tip: run with `-- --predictions /path/to.csv` or paste path here.")
        if not csv_path:
            st.stop()

        if not Path(csv_path).exists():
            st.error(f"File not found: {csv_path}")
            st.stop()

        df = load_predictions(csv_path)

        st.divider()
        st.header("Filters")

        # Build filter options
        y_true_opts = (
            sorted(df["y_true"].dropna().unique().tolist()) if "y_true" in df.columns else []
        )
        y_pred_opts = (
            sorted(df["y_pred"].dropna().unique().tolist()) if "y_pred" in df.columns else []
        )
        region_opts = (
            sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
        )
        year_opts = (
            sorted([int(x) for x in df["year"].dropna().unique().tolist()])
            if "year" in df.columns
            else []
        )

        y_true_sel = st.multiselect("y_true", options=y_true_opts, default=[])
        y_pred_sel = st.multiselect("y_pred", options=y_pred_opts, default=[])
        region_sel = st.multiselect("region", options=region_opts, default=[])
        year_sel = st.multiselect("year", options=year_opts, default=[])

        st.divider()
        st.header("Ordering")
        order_mode = st.radio(
            "Order",
            options=["random", "date", "region", "probability"],
            index=0,
            horizontal=True,
        )
        seed = st.number_input("Random seed", value=0, step=1, disabled=(order_mode != "random"))

        prob_cols_all = _find_prob_cols(df)
        sort_prob_col = None
        sort_prob_desc = True
        if order_mode == "probability":
            if prob_cols_all:
                sort_prob_col = st.selectbox(
                    "Sort by probability column",
                    options=prob_cols_all,
                    index=0,
                )
                sort_prob_desc = st.checkbox("Descending", value=True)
            else:
                st.caption("No probability columns found (expected columns starting with 'p_').")

        st.divider()
        st.header("Display")
        show_probs = st.checkbox("Show probability columns (p_*)", value=True)
        show_table = st.checkbox("Show row as table", value=False)

        # --- Image clipping controls (percentiles per band; always 3 bands) ---
        clip_preset = st.selectbox(
            "Image clipping preset",
            options=["2-98 (default)", "1-99", "5-95", "0-100 (none)"],
            index=0,
            help="Percentiles used to clip each of the 3 displayed bands before normalization.",
        )
        preset_map = {
            "2-98 (default)": (2, 98),
            "1-99": (1, 99),
            "5-95": (5, 95),
            "0-100 (none)": (0, 100),
        }
        preset_low, preset_high = preset_map[clip_preset]

        # Persist per-band values across reruns
        st.session_state.setdefault("clip_p_low", [preset_low, preset_low, preset_low])
        st.session_state.setdefault("clip_p_high", [preset_high, preset_high, preset_high])

        # Only apply preset when it changes (don't overwrite manual edits every rerun)
        last_preset = st.session_state.get("clip_preset_last")
        if last_preset != clip_preset:
            st.session_state["clip_p_low"] = [preset_low, preset_low, preset_low]
            st.session_state["clip_p_high"] = [preset_high, preset_high, preset_high]
            st.session_state["clip_preset_last"] = clip_preset

            # Keep the widget values synced on preset change
            for b in range(3):
                st.session_state[f"clip_low_{b}"] = int(preset_low)
                st.session_state[f"clip_high_{b}"] = int(preset_high)

        with st.expander("Advanced: per-band clipping", expanded=False):
            st.caption(
                "Percentiles are applied to each of the 3 displayed bands. "
                "Keep p_low < p_high. Integers in [0, 100]."
            )
            for b in range(3):
                c1, c2 = st.columns(2)
                low_key = f"clip_low_{b}"
                high_key = f"clip_high_{b}"

                st.session_state.setdefault(low_key, int(st.session_state["clip_p_low"][b]))
                st.session_state.setdefault(high_key, int(st.session_state["clip_p_high"][b]))

                with c1:
                    st.number_input(
                        f"Band {b} p_low",
                        min_value=0,
                        max_value=99,
                        value=int(st.session_state[low_key]),
                        step=1,
                        key=low_key,
                    )
                with c2:
                    st.number_input(
                        f"Band {b} p_high",
                        min_value=1,
                        max_value=100,
                        value=int(st.session_state[high_key]),
                        step=1,
                        key=high_key,
                    )

                # Enforce p_low < p_high
                if int(st.session_state[low_key]) >= int(st.session_state[high_key]):
                    st.session_state[high_key] = min(100, int(st.session_state[low_key]) + 1)

                st.session_state["clip_p_low"][b] = int(st.session_state[low_key])
                st.session_state["clip_p_high"][b] = int(st.session_state[high_key])

        # Final tuples passed into load_rgb_image()
        p_low = tuple(int(x) for x in st.session_state["clip_p_low"])
        p_high = tuple(int(x) for x in st.session_state["clip_p_high"])

    # Filter + sequence
    filtered = make_filtered_df(
        df,
        y_true_vals=y_true_sel,
        y_pred_vals=y_pred_sel,
        regions=region_sel,
        years=year_sel,
    )

    # Build a stable key for the current sequence definition
    sort_col_str = "" if sort_prob_col is None else sort_prob_col
    seq_key = (
        f"{csv_path}|{order_mode}|{seed}|"
        f"sortcol={sort_col_str}|desc={int(bool(sort_prob_desc))}|"
        f"ytrue={','.join(y_true_sel)}|ypred={','.join(y_pred_sel)}|"
        f"region={','.join(region_sel)}|year={','.join(map(str, year_sel))}"
    )
    if st.session_state["seq_key"] != seq_key:
        reset_sequence(seq_key)

    # Determine sequence mode
    if order_mode == "date":
        seq_mode = "date"
    elif order_mode == "region":
        seq_mode = "region"
    elif order_mode == "probability" and sort_prob_col is not None:
        seq_mode = f"col:{sort_prob_col}"
    else:
        seq_mode = "random"

    seq = build_sequence(filtered, mode=seq_mode, seed=int(seed))

    # If sorting by probability and user requested ascending, reverse the sorted order
    if order_mode == "probability" and sort_prob_col is not None and not sort_prob_desc:
        seq = seq.iloc[::-1]

    if seq.empty:
        st.warning("No rows match current filters.")
        st.stop()

    # Clamp index
    st.session_state["idx"] = int(np.clip(st.session_state["idx"], 0, len(seq) - 1))

    # Navigation callbacks: update state BEFORE Streamlit reruns the script
    def _prev_cb():
        st.session_state["idx"] = max(0, int(st.session_state["idx"]) - 1)
        st.session_state["jump_idx"] = st.session_state["idx"]

    def _next_cb():
        st.session_state["idx"] = min(len(seq) - 1, int(st.session_state["idx"]) + 1)
        st.session_state["jump_idx"] = st.session_state["idx"]

    def _go_cb():
        st.session_state["idx"] = int(
            np.clip(int(st.session_state.get("jump_idx", 0)), 0, len(seq) - 1)
        )
        st.session_state["jump_idx"] = st.session_state["idx"]

    row = seq.iloc[st.session_state["idx"]]
    tif_path = str(row["source_tif"])

    # Main layout: image left, metadata right
    left, right = st.columns([3, 2], gap="large")

    with left:
        if not Path(tif_path).exists():
            st.error(f"Missing tif: {tif_path}")
        else:
            try:
                rgb = load_rgb_image(tif_path, p_low=p_low, p_high=p_high)
                st.image(rgb, caption=os.path.basename(tif_path), width="stretch")
            except Exception as e:
                st.exception(e)

    with right:
        # Navigation controls (moved to right so the image stays at the top)
        nav1, nav2 = st.columns([1, 1])
        with nav1:
            st.button(
                "⟵ Prev",
                use_container_width=True,
                disabled=(st.session_state["idx"] <= 0),
                on_click=_prev_cb,
            )
        with nav2:
            st.button(
                "Next ⟶",
                use_container_width=True,
                disabled=(st.session_state["idx"] >= len(seq) - 1),
                on_click=_next_cb,
            )

        jump_cols = st.columns([2, 1])
        with jump_cols[0]:
            # Keep jump input separate so it can change without immediately changing idx
            st.session_state["jump_idx"] = int(
                np.clip(
                    int(st.session_state.get("jump_idx", st.session_state["idx"])), 0, len(seq) - 1
                )
            )
            st.number_input(
                "Jump to index",
                min_value=0,
                max_value=len(seq) - 1,
                step=1,
                key="jump_idx",
            )
        with jump_cols[1]:
            st.button("Go", use_container_width=True, on_click=_go_cb)

        st.divider()

        st.subheader(f"Image {st.session_state['idx'] + 1} / {len(seq)}")
        # Show asset_id prominently (easy to copy into Label Studio filters)
        asset_id = None
        if "asset_id" in row.index and pd.notna(row["asset_id"]):
            asset_id = str(row["asset_id"])

        # Keep the text input value synced to the currently displayed row.
        # IMPORTANT: don't pass both `value=` and set session_state for the same key.
        if asset_id is not None:
            st.session_state.setdefault("asset_id_copy", "")
            st.session_state["asset_id_copy"] = asset_id
            st.text_input(
                "asset_id (copy)",
                key="asset_id_copy",
                help="Copy/paste this into Label Studio to filter and fix labels.",
            )
        else:
            st.caption("asset_id: (missing)")

        st.divider()

        st.subheader("Metadata")
        meta_fields = ["asset_id", "region", "year", "acquired", "y_pred", "y_true", "source_tif"]
        present = [c for c in meta_fields if c in row.index]
        st.json({c: (None if pd.isna(row[c]) else str(row[c])) for c in present})

        if show_probs:
            prob_cols = _find_prob_cols(df)
            if prob_cols:
                st.subheader("Probabilities")
                probs = {c: float(row[c]) for c in prob_cols if c in row.index and pd.notna(row[c])}
                if probs:
                    # Show sorted high-to-low
                    probs_sorted = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
                    st.table(
                        pd.DataFrame(
                            {"class": list(probs_sorted.keys()), "p": list(probs_sorted.values())}
                        )
                    )
                else:
                    st.caption("No p_* values present for this row.")
            else:
                st.caption("No probability columns found (expected columns starting with 'p_').")

        if show_table:
            st.subheader("Row (raw)")
            st.dataframe(pd.DataFrame(row).T)

    st.caption(
        "Filters apply before ordering. Date ordering uses `acquired` if present. "
        "Random ordering is stable for a given seed + filter set."
    )


if __name__ == "__main__":
    main()
