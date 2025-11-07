import shutil
from pathlib import Path

import pandas as pd


def process_df(df: pd.DataFrame, out_name: str) -> pd.DataFrame:
    remove_idx = []
    for i, row in df.iterrows():
        source = Path(row.source_tif)
        if not source.exists():
            remove_idx.append(i)
            continue

        dove = source.parents[5].name
        year = source.parents[3].name
        month = source.parents[2].name
        region = source.parents[1].name
        out = (
            Path("/Users/kyledorman/data/estuary/dataset/images/")
            / dove
            / "results"
            / year
            / month
            / region
            / "files"
            / source.name
        )
        if not out.exists():
            out.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(source, out)
        df.loc[i, "source_tif"] = str(out)  # type: ignore

    df = df[~df.index.isin(remove_idx)].copy().reset_index(drop=True)
    df.to_csv(f"/Users/kyledorman/data/estuary/dataset/{out_name}.csv", index=False)

    return df


LOW_QUALITY_LABELS = Path("/Volumes/x10pro/estuary/low_quality/labeled_data.csv")
TIME_SERIES_LABELS = Path("/Volumes/x10pro/estuary/ca_all/dove/labels.csv")
TRAIN_LABELS = Path("/Volumes/x10pro/estuary/dove/labels.csv")
VALID_REGIONS = [int(p.stem) for p in Path("/Volumes/x10pro/estuary/ca_grids").glob("*.geojson")]

lq_df = pd.read_csv(LOW_QUALITY_LABELS)
ts_df = pd.read_csv(TIME_SERIES_LABELS)
train_df = pd.read_csv(TRAIN_LABELS)

time_series_regions = ts_df.region.unique()
print("Time Series Regions")
print(time_series_regions.tolist())

# Move low quality labels for time series regions to time series dataset
ts_df["dup_order"] = 1
lq_df["dup_order"] = 0
ts_df = (
    pd.concat([ts_df, lq_df[lq_df.region.isin(time_series_regions)]])
    .sort_values(by=["region", "acquired"])
    .reset_index(drop=True)
    .drop_duplicates(["source_tif"])
    .drop(columns=["dup_order"])
    .reset_index(drop=True)
)
ts_df["source"] = "time_series"

lq_df = (
    lq_df[~lq_df.region.isin(time_series_regions)]
    .sort_values(by=["region", "acquired"])
    .reset_index(drop=True)
    .drop(columns=["dup_order"])
)

lq_df = process_df(lq_df, "low_quality")
ts_df = process_df(ts_df, "time_series")
train_df = process_df(train_df, "train_dataset")

lq_df["dup_order"] = 0
lq_df["source"] = "low_quality"
train_df["dup_order"] = 1
train_df["source"] = "train"

merged_df = pd.concat([lq_df, train_df])
merged_df = (
    merged_df.sort_values(["source_tif", "dup_order"])
    .reset_index(drop=True)
    .drop_duplicates(["source_tif"])
    .drop(columns=["dup_order"])
    .reset_index(drop=True)
)
merged_df.to_csv("/Users/kyledorman/data/estuary/dataset/train_low_quality.csv", index=False)
