import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from estuary.model.data import EstuaryDataModule, EstuaryDataset, num_workers
from estuary.model.module import EstuaryModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    module = EstuaryModule.load_from_checkpoint(args.model_path, batch_size=1).eval()
    module = module.eval()

    dm = EstuaryDataModule(module.conf)
    dm.prepare_data()
    dm.setup()

    assert dm.val_aug is not None
    assert dm.train_aug is not None

    assert not module.conf.cv_folds, "Remove test_ds below"
    dfs = pd.concat([dm.train_ds.df, dm.val_ds.df, dm.test_ds.df], ignore_index=True)  # type: ignore
    ds = EstuaryDataset(
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
    if module.conf.split_method == "crossval":
        fold = module.conf.cv_index
    else:
        fold = -1

    save_path = Path(args.model_path).parent.parent / "preds.csv"
    results = []
    for batch in tqdm.tqdm(dl, total=len(dl)):
        batch = ds.transforms(batch)
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
                    "y_true": batch["label"][i].detach().cpu().numpy(),
                    "y_prob": probs_pos[i].detach().cpu().numpy()[0],
                    "y_pred": (probs_pos[i] > 0.5).to(torch.int32).detach().cpu().numpy()[0],
                    "region": int(Path(batch["source_tif"][i]).parents[2].name),
                    "fold": fold,
                }
            )

    results_df = pd.DataFrame(results)
    results_df = pd.merge(
        results_df, ds.df[["source_tif", "orig_label", "dataset"]], on="source_tif", how="left"
    )
    results_df["correct"] = results_df.y_true == results_df.y_pred
    if save_path.exists():
        os.remove(save_path)
    results_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
