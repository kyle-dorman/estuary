from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def build_final_preds(
    df: pd.DataFrame, threshold_main: float, threshold_unsure: float
) -> np.ndarray:
    """
    Final label encoding:
      0 = closed
      1 = open
      2 = unsure
    """
    # binary main model
    if "p_open" not in df.columns:
        raise ValueError("Expected column 'p_open' in predictions CSV.")

    y_pred_main = (df["p_open"].to_numpy() >= threshold_main).astype(int)
    y_pred_unsure = df["y_prob_unsure"].to_numpy() >= threshold_unsure

    y_pred_final = y_pred_main.copy()
    y_pred_final[y_pred_unsure] = 2
    return y_pred_final


def build_final_truth(df: pd.DataFrame) -> np.ndarray:
    """
    Maps ground truth into:
      0 = closed
      1 = open
      2 = unsure
    Assumes:
      y_true in {0,1} for sure labels
      y_true == -1 for unsure
    """
    y_true = df["y_true"].to_numpy().copy()
    y_true_final = y_true.copy()
    y_true_final[y_true_final < 0] = 2
    return y_true_final


def summarize_thresholds(
    df: pd.DataFrame,
    threshold_main_values: np.ndarray,
    threshold_unsure_values: np.ndarray,
) -> pd.DataFrame:
    y_true_final = build_final_truth(df)

    rows = []
    for t_main in threshold_main_values:
        for t_unsure in threshold_unsure_values:
            y_pred_final = build_final_preds(df, t_main, t_unsure)

            macro_f1 = f1_score(
                y_true_final,
                y_pred_final,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )

            f1_closed = f1_score(
                y_true_final == 0,
                y_pred_final == 0,
                zero_division=0,
            )
            f1_open = f1_score(
                y_true_final == 1,
                y_pred_final == 1,
                zero_division=0,
            )
            f1_unsure = f1_score(
                y_true_final == 2,
                y_pred_final == 2,
                zero_division=0,
            )

            rows.append(
                {
                    "threshold_main": round(float(t_main), 4),
                    "threshold_unsure": round(float(t_unsure), 4),
                    "macro_f1": float(macro_f1),
                    "f1_closed": float(f1_closed),
                    "f1_open": float(f1_open),
                    "f1_unsure": float(f1_unsure),
                }
            )

    out = pd.DataFrame(rows).sort_values(
        ["macro_f1", "f1_unsure", "f1_open", "f1_closed"],
        ascending=False,
    )
    return out


@click.command()
@click.option(
    "--preds-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="CSV of model predictions.",
)
@click.option(
    "--save-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    default=None,
    help="Directory to save outputs. Defaults to preds CSV parent.",
)
@click.option("--main-min", type=float, default=0.05)
@click.option("--main-max", type=float, default=0.95)
@click.option("--main-step", type=float, default=0.01)
@click.option("--unsure-min", type=float, default=0.05)
@click.option("--unsure-max", type=float, default=0.95)
@click.option("--unsure-step", type=float, default=0.01)
def main(
    preds_path: Path,
    save_dir: Path | None,
    main_min: float,
    main_max: float,
    main_step: float,
    unsure_min: float,
    unsure_max: float,
    unsure_step: float,
):
    df = pd.read_csv(preds_path)
    df = df[df.year == 2021]

    required_cols = {"y_true", "y_prob_unsure", "p_open"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if save_dir is None:
        save_dir = preds_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    threshold_main_values = np.arange(main_min, main_max + 1e-9, main_step)
    threshold_unsure_values = np.arange(unsure_min, unsure_max + 1e-9, unsure_step)

    results = summarize_thresholds(
        df=df,
        threshold_main_values=threshold_main_values,
        threshold_unsure_values=threshold_unsure_values,
    )

    best = results.iloc[0]

    print("\nBest thresholds")
    print("================")
    print(f"main threshold   : {best['threshold_main']:.3f}")
    print(f"unsure threshold : {best['threshold_unsure']:.3f}")
    print(f"macro F1         : {best['macro_f1']:.4f}")
    print(f"closed F1        : {best['f1_closed']:.4f}")
    print(f"open F1          : {best['f1_open']:.4f}")
    print(f"unsure F1        : {best['f1_unsure']:.4f}")

    results_path = save_dir / f"{preds_path.stem}_threshold_sweep.csv"
    results.to_csv(results_path, index=False)

    best_path = save_dir / f"{preds_path.stem}_best_thresholds.csv"
    pd.DataFrame([best]).to_csv(best_path, index=False)

    print("\nSaved:")
    print(results_path)
    print(best_path)


if __name__ == "__main__":
    main()
