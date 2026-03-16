import os
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from estuary.low_quality.module import LowQualityModule
from estuary.model.data import EstuaryDataset
from estuary.model.module import EstuaryModule
from estuary.util.data import parse_dt_from_pth


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
    "--unsure-model-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--threshold-unsure", type=float, default=0.5)
@click.option("--threshold", type=float, default=0.5)
def main(
    labels_path: Path,
    model_path: Path,
    unsure_model_path: Path,
    threshold_unsure: float,
    threshold: float,
):
    save_path = model_path.parent.parent / f"{labels_path.stem}_preds.csv"
    if save_path.exists():
        return

    module = EstuaryModule.load_from_checkpoint(model_path, batch_size=1, weights_only=False).eval()
    unsure_module = LowQualityModule.load_from_checkpoint(
        unsure_model_path, batch_size=1, weights_only=False
    ).eval()

    labels = pd.read_csv(labels_path)
    if "label" not in labels.columns:
        labels["label"] = "open"
    labels["acquired"] = labels["source_tif"].apply(lambda p: parse_dt_from_pth(Path(p)))
    labels["date"] = labels["acquired"].dt.date  # type: ignore
    labels["orig_label"] = labels["label"]
    # build a lookup dict → index
    lookup = {tok: idx for idx, tok in enumerate(module.conf.classes)}
    lookup["unsure"] = -1
    if "perched open" not in module.conf.classes:
        labels.loc[labels.label == "perched open", "label"] = "open"
    labels["label_idx"] = labels["label"].map(lookup)
    labels["y_true"] = labels.label_idx
    labels["y_true_unsure"] = labels.label.apply(lambda a: int(a == "unsure"))

    gdf = gpd.read_file(Path("/Volumes/x10pro/estuary/") / "geos/ca_data_w_empa_pmep_usgs.geojson")
    valid_regions = gdf[~gdf.skipped]["Site code"].tolist()

    labels = labels[labels.region.isin(valid_regions)].copy()

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

        logits: torch.Tensor = module(batch)

        row = {}

        if len(module.conf.classes) > 2:
            probs = torch.softmax(logits, dim=1)
            probs_np: np.ndarray = probs.detach().cpu().numpy()[0]

            # Predicted class + confidence
            y_pred = probs.argmax(dim=1).item()
            y_prob = probs[0, y_pred].item()  # type: ignore

            for cls, p in zip(module.conf.classes, probs_np, strict=True):
                row[f"p_{cls.replace(' ', '_')}"] = float(p)
        else:
            probs = torch.sigmoid(logits)
            probs_np = probs.detach().cpu().numpy()[0, 0].item()

            y_pred = int(probs_np > threshold)
            if y_pred:
                y_prob = probs_np
            else:
                y_prob = 1.0 - probs_np

            for i, cls in enumerate(module.conf.classes):
                if i == 1:
                    row[f"p_{cls.replace(' ', '_')}"] = float(probs_np)
                else:
                    row[f"p_{cls.replace(' ', '_')}"] = 1.0 - float(probs_np)

        row.update(
            {
                "source_tif": batch["source_tif"][0],
                "y_prob": y_prob,
                "y_prob_unsure": unsure_probs,
                "y_pred": y_pred,
                "y_pred_unsure": unsure_probs > threshold_unsure,
            }
        )

        results_list.append(row)

    results_df = pd.DataFrame(results_list)
    results_df = pd.merge(results_df, labels, on="source_tif", how="left")

    if len(results_df.y_true.unique()) > 1:
        print("Results", model_path.parent.parent.name)
        print("==========================")

        print("F1 w/sure")
        no_unsure = results_df[results_df.y_true >= 0]
        print(f1_score(no_unsure.y_true, no_unsure.y_pred, average="macro"))

        print("F1 w/sure & pred sure")
        no_unsure2 = results_df[(results_df.y_true >= 0) & (results_df.y_pred_unsure == 0)]
        print(f1_score(no_unsure2.y_true, no_unsure2.y_pred, average="macro"))

        print("F1 unsure")
        print(f1_score(results_df.y_true_unsure, results_df.y_pred_unsure))

    if save_path.exists():
        os.remove(save_path)
    results_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
