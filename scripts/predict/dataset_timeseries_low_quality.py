import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from estuary.low_quality.data import LowQualityDataModule, LowQualityDataset, num_workers
from estuary.low_quality.module import LowQualityModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    module = LowQualityModule.load_from_checkpoint(
        args.model_path,
        batch_size=1,
        strict=False,
    ).eval()
    module = module.eval()

    dm = LowQualityDataModule(module.conf)
    dm.prepare_data()
    dm.setup()

    assert dm.val_aug is not None

    dfs = pd.concat([dm.train_ds.df, dm.val_ds.df, dm.test_ds.df], ignore_index=True)  # type: ignore
    ds = LowQualityDataset(
        df=dfs,
        conf=module.conf,
        train=False,
    )
    dl = DataLoader(
        ds,
        batch_size=module.conf.batch_size,
        shuffle=False,
        num_workers=num_workers(module.conf),
        pin_memory=module.conf.pin_memory,
        persistent_workers=module.conf.persistent_workers,
    )

    train_end = len(dm.train_ds.df)  # type: ignore
    end_val = train_end + len(dm.val_ds.df)  # type: ignore

    results = []
    for bi, batch in tqdm.tqdm(enumerate(dl), total=len(dl)):
        if bi < train_end:
            dataset = "train"
        elif bi < end_val:
            dataset = "val"
        else:
            dataset = "test"
        batch = dm.val_aug(batch)
        for k in batch.keys():
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(module.device)
        logits = module.forward(batch)
        probs_pos = torch.sigmoid(logits)

        for i in range(len(probs_pos)):
            results.append(
                {
                    "source_tif": batch["source_tif"][i],
                    "dataset": dataset,
                    "y_true": batch["label"][i].detach().cpu().numpy().item(),
                    "y_prob": probs_pos[i].detach().cpu().numpy()[0],
                }
            )

    results_df = pd.DataFrame(results)
    results_df = pd.merge(results_df, ds.df, on=["source_tif", "dataset"], how="left")

    thresholds = np.linspace(0, 1, 101)  # e.g., 0.00, 0.01, ..., 1.00
    scores = []

    y_prob = results_df[results_df.dataset == "train"].y_prob
    y_true = results_df[results_df.dataset == "train"].y_true

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        scores.append((t, f1, acc, prec, rec))

    # put into DataFrame for analysis
    df_scores = pd.DataFrame(scores, columns=["threshold", "f1", "accuracy", "precision", "recall"])

    # best threshold by F1
    best_f1_row = df_scores.loc[df_scores["f1"].idxmax()]
    print(best_f1_row)

    threshold = best_f1_row.threshold
    results_df["y_pred"] = (results_df.y_prob > threshold).astype(np.int32)
    results_df["correct"] = (results_df.y_true == results_df.y_pred).astype(np.int32)

    save_path = Path(args.model_path).parent.parent / "preds.csv"
    if save_path.exists():
        os.remove(save_path)
    results_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
