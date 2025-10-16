import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from estuary.model.data import EstuaryDataset, _load_labels
from estuary.model.module import EstuaryModule


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
)
@click.option(
    "--save-path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), required=False
)
def main(labels_path: Path, model_path: Path, save_path: Path | None):
    if save_path is None:
        save_path = model_path.parent.parent / "timeseries_preds.csv"
    module = EstuaryModule.load_from_checkpoint(model_path, batch_size=1).eval()

    labels = _load_labels(module.conf.classes, labels_path)

    ds = EstuaryDataset(labels, module.conf, train=False)
    dl = DataLoader(
        ds,
        batch_size=module.conf.batch_size,
    )
    results_list = []
    for batch in tqdm.tqdm(dl, total=len(labels)):
        batch = ds.transforms(batch)
        for k in batch.keys():
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(module.device)

        logits = module(batch)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs > 0.5)[0].astype(np.int32).item()
            probs = probs[0].item()
        else:
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1).tolist()[0].item()
            probs = probs[0, 0].item()

        results_list.append(
            {
                "source_tif": batch["source_tif"][0],
                "y_true": batch["label"][0].detach().cpu().numpy(),
                "y_prob": probs,
                "y_pred": preds,
                "region": int(Path(batch["source_tif"][0]).parents[1].name),
                "dataset": "train",
            }
        )

    ca_results_df = pd.DataFrame(results_list)
    ca_results_df = pd.merge(
        ca_results_df, labels[["source_tif", "orig_label"]], on="source_tif", how="left"
    )
    ca_results_df["correct"] = ca_results_df.y_true == ca_results_df.y_pred
    if save_path.exists():
        os.remove(save_path)
    ca_results_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
