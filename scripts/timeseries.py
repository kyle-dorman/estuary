import json
from pathlib import Path

import pandas as pd
import rasterio
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from estuary.clay.data import EstuaryDataset, parse_dt_from_pth
from estuary.clay.module import EstuaryModule

BASE = Path("/Users/kyledorman/data/estuary/dove/results")
LABELS_PATH = Path("/Users/kyledorman/data/estuary/label_studio/00025/labels.csv")
CROP_PATH = Path("/Users/kyledorman/data/estuary/label_studio/region_crops.json")
VALID_PATH = Path("/Users/kyledorman/data/estuary/valid.csv")
SAVE_PATH = Path("/Users/kyledorman/data/estuary/preds.csv")
MODEL_PATH = Path(
    "/Users/kyledorman/data/results/estuary/train/20250805-205230/checkpoints/last.ckpt"
)

MIN_CLEAR = 0.90


def main():
    labels = pd.read_csv(LABELS_PATH)
    regions = sorted(list(set(labels.region.tolist())))
    region_crops = json.loads(CROP_PATH.read_bytes())

    if not VALID_PATH.exists():
        valid_files = []
        for region in tqdm.tqdm(regions):
            crop = region_crops[region]
            start_w, start_h, end_w, end_h = crop
            w = end_w - start_w
            h = end_h - start_h
            udms = list(BASE.glob(f"*/*/{region}/files/*_udm2_clip.tif"))

            for udm in udms:
                with rasterio.open(udm) as src:
                    clear = src.read(1) == 1
                    clear = clear[start_h:end_h, start_w:end_w]
                    pct_clear = clear.sum() / w / h
                    if pct_clear < MIN_CLEAR:
                        continue

                capture_id = "_".join(udm.stem.split("_")[:4])
                tif_path = list(udm.parent.glob(f"{capture_id}*AnalyticMS_SR*.tif"))[0]

                with rasterio.open(tif_path) as src:
                    no_data = src.read(1, masked=True).mask[start_h:end_h, start_w:end_w]
                    yes_data_pct = (~no_data).sum() / w / h
                    if yes_data_pct < MIN_CLEAR:
                        continue

                valid_files.append((region, udm, tif_path, 0, 0, 0.0))

        valid_df = pd.DataFrame(
            valid_files, columns=["region", "udm_path", "source_tif", "label_idx", "pred", "conf"]
        )
        valid_df.to_csv(VALID_PATH, index=False)
    else:
        valid_df = pd.read_csv(VALID_PATH)

    valid_df["acquired"] = valid_df.source_tif.apply(lambda a: parse_dt_from_pth(Path(a)))
    valid_df["acquired"] = pd.to_datetime(valid_df["acquired"], errors="coerce")
    valid_df = valid_df.sort_values(by=["region", "acquired"]).reset_index()
    # valid_df = valid_df[valid_df.region == "big_sur_river"]
    # valid_df = valid_df.copy(deep=True)
    # valid_df = valid_df.reset_index()

    module = EstuaryModule.load_from_checkpoint(MODEL_PATH, batch_size=1).eval()
    module.conf.holdout_region = None
    ds = EstuaryDataset(valid_df, region_crops, module.conf, None, train=False)
    dl = DataLoader(
        ds,
        batch_size=module.conf.batch_size,
    )
    for bi, batch in tqdm.tqdm(enumerate(dl), total=len(valid_df)):
        # Unpack dataset output: either a dict or (dict, label)
        if isinstance(batch, list | tuple):
            datacube = batch[0]
        else:
            datacube = batch

        # Move tensors to the module's device
        for k, v in list(datacube.items()):
            if hasattr(v, "to"):
                datacube[k] = v.to(module.device)

        logits = module(datacube)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        valid_df.loc[bi, "pred"] = int(probs.argmax(axis=1).tolist()[0])
        valid_df.loc[bi, "conf"] = probs[0, 0].item()

    valid_df.to_csv(SAVE_PATH, index=False)


if __name__ == "__main__":
    main()
