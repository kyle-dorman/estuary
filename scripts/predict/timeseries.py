import os
from pathlib import Path

import click
import pandas as pd
import torch
import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from estuary.low_quality.module import LowQualityModule
from estuary.model.data import EstuaryDataset
from estuary.model.module import EstuaryModule
from estuary.util.data import parse_dt_from_pth


def label_map(label: str) -> str:
    if label == "perched open":
        return "open"
    return label


@click.command()
@click.option(
    "--labels-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)  # 20251203-095017
@click.option(
    "--unsure-model-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--threshold", type=float, default=0.5)
@click.option("--threshold-unsure", type=float, default=0.5)
def main(
    labels_path: Path,
    model_path: Path,
    unsure_model_path: Path,
    threshold: float,
    threshold_unsure: float,
):
    save_path = model_path.parent.parent / f"{labels_path.stem}_preds.csv"

    module = EstuaryModule.load_from_checkpoint(model_path, batch_size=1).eval()
    unsure_module = LowQualityModule.load_from_checkpoint(unsure_model_path, batch_size=1).eval()

    labels = pd.read_csv(labels_path)
    labels["acquired"] = labels["source_tif"].apply(lambda p: parse_dt_from_pth(Path(p)))
    labels["date"] = labels["acquired"].dt.date
    labels["orig_label"] = labels["label"]
    labels["label_idx"] = labels.label.apply(lambda a: int(a != "closed"))
    labels["y_true"] = labels.label.apply(lambda a: int(a != "closed"))
    labels["y_true_unsure"] = labels.label.apply(lambda a: int(a == "unsure"))

    ds = EstuaryDataset(labels, module.conf, train=False)
    dl = DataLoader(
        ds,
        batch_size=module.conf.batch_size,
    )
    results_list = []
    for batch in tqdm.tqdm(dl, total=len(labels)):
        for k in batch.keys():
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(module.device)

        unsure_logits = unsure_module(batch)
        unsure_probs = torch.sigmoid(unsure_logits).detach().cpu().numpy()
        unsure_probs = unsure_probs[0].item()

        logits = module(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs = probs[0].item()

        results_list.append(
            {
                "source_tif": batch["source_tif"][0],
                "y_prob": probs,
                "y_prob_unsure": unsure_probs,
                "y_pred": probs > threshold,
                "y_pred_unsure": unsure_probs > threshold_unsure,
            }
        )

    results_df = pd.DataFrame(results_list)
    results_df = pd.merge(results_df, labels, on="source_tif", how="left")

    print("Results", model_path.parent.parent.name)
    print("==========================")
    print("F1 w/sure")
    print(f1_score(results_df.y_true, results_df.y_pred))
    print("F1 unsure")
    print(f1_score(results_df.y_true_unsure, results_df.y_pred_unsure))

    if save_path.exists():
        os.remove(save_path)
    results_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
