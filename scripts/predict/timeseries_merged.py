import os
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import tqdm

from estuary.model.config import CLASSES
from estuary.util.data import parse_dt_from_pth


def label_score(label: str) -> int | None:
    return {
        "closed": 0,
        "perched open": 1,
        "open": 1,
        "unsure": None,
    }[label]


def grouped_label(df: pd.DataFrame) -> tuple[int, int]:
    # Take the mean of the labels, treating "perched open" as a 50/50 prediction
    # Tie goes to open
    df["label_score"] = df.label.apply(label_score)
    clean_df = df[df.y_true_unsure == 0]

    if len(clean_df) == 0:
        return -1, 1

    y_true = int(clean_df.label_score.mean() >= 1.0)

    return y_true, 0


def grouped_source_tif(df: pd.DataFrame) -> str:
    clean_df = df[df.y_true_unsure == 0]

    if len(clean_df) == 0:
        return df.iloc[0]["source_tif"]
    return clean_df.iloc[0]["source_tif"]


def weights_mean(df: pd.DataFrame, key: str, weight_key: str) -> float:
    """Compute a weighted mean of `df[key]` using weights derived from `weight_key`.

    We interpret `df[weight_key]` as an *uncertainty* (higher = worse), so the weight is
    `w = 1 - df[weight_key]`. Values with NaNs in either column are ignored.

    Returns NaN if no valid weights exist or if the total weight is zero.
    """
    x = pd.to_numeric(df[key], errors="coerce")
    w = 1.0 - pd.to_numeric(df[weight_key], errors="coerce")

    m = x.notna() & w.notna()
    if not m.any():
        return float("nan")

    x = x[m]
    w = w[m]

    # Clamp weights into [0, 1] to avoid negative weights from bad inputs
    w = w.clip(lower=0.0, upper=1.0)

    wsum = float(w.sum())
    if wsum <= 0:
        return float("nan")

    return float((x * w).sum() / wsum)


def grouped_prediction(
    df: pd.DataFrame, threshold_unsure: float, classes: list[str]
) -> dict[str, Any]:
    clean_df = df[df.y_prob_unsure < threshold_unsure]

    if len(clean_df) == 0:
        return {"y_pred_unsure": 1}
    elif len(clean_df) == 1:
        row = clean_df.iloc[0]
        out = {"y_pred": row.y_pred, "openness": row.openness, "y_pred_unsure": 0}
        for cls in classes:
            cls = cls.replace(" ", "_")
            key = f"p_{cls}"
            out[key] = row[key]
        return out

    out: dict[str, Any] = {"y_pred_unsure": 0}

    scores = []
    for cls in classes:
        cls = cls.replace(" ", "_")
        key = f"p_{cls}"
        score = weights_mean(clean_df, key, "y_prob_unsure")
        scores.append(score)

    # Normalize scores
    row_sum = np.nansum(scores, keepdims=True)
    scores_norm = np.array(scores) / row_sum
    for i, cls in enumerate(classes):
        cls = cls.replace(" ", "_")
        key = f"p_{cls}"
        out[key] = scores_norm[i]

    out["openness"] = weights_mean(clean_df, "openness", "y_prob_unsure")
    out["y_pred"] = np.argmax(scores)

    return out


def region_balanced_metric(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    metric_fn,
) -> float:
    """Compute a region-balanced metric by averaging per-region scores.

    Regions with only a single class present in y_true are skipped.
    """
    per_region_scores: list[float] = []
    for _, rdf in df.groupby("region"):
        # need at least 2 classes to compute precision/recall meaningfully
        if rdf[y_true_col].nunique() < 2:
            continue
        score = metric_fn(rdf[y_true_col], rdf[y_pred_col])
        per_region_scores.append(score)

    if len(per_region_scores) == 0:
        return float("nan")

    return float(np.mean(per_region_scores))


@click.command()
@click.option(
    "--preds-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--threshold-unsure", type=float, default=0.5)
def main(
    preds_path: Path,
    threshold_unsure: float,
):
    save_path = preds_path.parent / f"merged_{preds_path.name}"

    results = pd.read_csv(preds_path)
    results["acquired"] = results["source_tif"].apply(lambda p: parse_dt_from_pth(Path(p)))
    results["date"] = results["acquired"].dt.date  # type: ignore

    merged_results = []
    for region, rdf in tqdm.tqdm(results.groupby("region"), total=len(results.region.unique())):
        for date, ddf in rdf.groupby("date"):
            y_true, y_true_unsure = grouped_label(ddf)
            pred = grouped_prediction(ddf, threshold_unsure, list(CLASSES))

            out = {
                "region": region,
                "date": date,
                "y_true": y_true,
                "y_true_unsure": y_true_unsure,
                "acquired": ddf.acquired.mean(),
                "source_tif": grouped_source_tif(ddf),
            }
            out.update(pred)

            merged_results.append(out)

    merged_df = pd.DataFrame(merged_results)

    # # Metrics for data quality filter
    # print("Merged Results")

    # unsure_recall = recall_score(merged_df.y_true_unsure, merged_df.y_pred_unsure)
    # print("Unsure Recall:", round(unsure_recall, 3))  # type: ignore
    # unsure_precision = precision_score(merged_df.y_true_unsure, merged_df.y_pred_unsure)
    # print("Unsure Precision:", round(unsure_precision, 3))  # type: ignore

    # unsure_recall_region = region_balanced_metric(
    #     merged_df, "y_true_unsure", "y_pred_unsure", recall_score
    # )
    # print("Unsure Recall (region-balanced):", round(unsure_recall_region, 3))  # type: ignore

    # unsure_precision_region = region_balanced_metric(
    #     merged_df, "y_true_unsure", "y_pred_unsure", precision_score
    # )
    # print("Unsure Precision (region-balanced):", round(unsure_precision_region, 3))  # type: ignore

    # # Metrics for only known good data
    # open_df = merged_df[merged_df.y_true_unsure == 0].copy()
    # open_df["y_true"] = (open_df.y_true == 1).astype(np.int32)
    # open_df["y_pred"] = (open_df.y_pred == 1).astype(np.int32)

    # open_recall = recall_score(open_df.y_true, open_df.y_pred)
    # print("Open Recall:", round(open_recall, 3))  # type: ignore
    # open_precision = precision_score(open_df.y_true, open_df.y_pred)
    # print("Open Precision:", round(open_precision, 3))  # type: ignore

    # open_recall_region = region_balanced_metric(open_df, "y_true", "y_pred", recall_score)
    # print("Open Recall (region-balanced):", round(open_recall_region, 3))  # type: ignore

    # open_precision_region = region_balanced_metric(open_df, "y_true", "y_pred", precision_score)
    # print("Open Precision (region-balanced):", round(open_precision_region, 3))  # type: ignore

    # # Merge predictions (both must be correct)
    # merged_df["y_true_total"] = ((merged_df.y_true == 1) & (merged_df.y_true_unsure == 0)).astype(
    #     np.int32
    # )
    # merged_df["y_pred_total"] = ((merged_df.y_pred == 1) & (merged_df.y_pred_unsure == 0)).astype(
    #     np.int32
    # )
    # total_recall = recall_score(merged_df.y_true_total, merged_df.y_pred_total)
    # print("Total Recall:", round(total_recall, 3))  # type: ignore
    # total_precision = precision_score(merged_df.y_true_total, merged_df.y_pred_total)
    # print("Total Precision:", round(total_precision, 3))  # type: ignore

    # total_recall_region = region_balanced_metric(
    #     merged_df, "y_true_total", "y_pred_total", recall_score
    # )
    # print("Total Recall (region-balanced):", round(total_recall_region, 3))  # type: ignore

    # total_precision_region = region_balanced_metric(
    #     merged_df, "y_true_total", "y_pred_total", precision_score
    # )
    # print("Total Precision (region-balanced):", round(total_precision_region, 3))  # type: ignore

    # # Merge predictions (both must be correct)
    # results["y_true_total"] = ((results.y_true == 1) & (results.y_true_unsure == 0)).astype(
    #     np.int32
    # )
    # results["y_pred_total"] = ((results.y_pred == 1) & (results.y_pred_unsure == 0)).astype(
    #     np.int32
    # )
    # total_recall = recall_score(results.y_true_total, results.y_pred_total)
    # print("Total Recall (no time series):", round(total_recall, 3))  # type: ignore
    # total_precision = precision_score(results.y_true_total, results.y_pred_total)
    # print("Total Precision (no time series):", round(total_precision, 3))  # type: ignore

    # total_recall_region = region_balanced_metric(
    #     results, "y_true_total", "y_pred_total", recall_score
    # )
    # print("Total Recall (region-balanced, no time series):", round(total_recall_region, 3))

    # total_precision_region = region_balanced_metric(
    #     results, "y_true_total", "y_pred_total", precision_score
    # )
    # print("Total Precision (region-balanced, no time series):", round(total_precision_region, 3))

    if save_path.exists():
        os.remove(save_path)
    merged_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
