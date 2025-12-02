import shutil
from pathlib import Path

import pandas as pd


def out_tif_path(row) -> Path:
    return (
        Path("/Users/kyledorman/data/estuary/dataset/images/")
        / row.dove
        / "results"
        / str(row.year)
        / str(row.month)
        / str(row.region)
        / "files"
        / Path(row.source_tif).name
    )


def process_df(df: pd.DataFrame, out_name: str) -> pd.DataFrame:
    print(out_name)

    keep = df.source_tif.apply(lambda a: Path(a).exists())
    print(df[~keep][["region", "year", "month", "dove"]])

    df["new_source_tif"] = df.apply(out_tif_path, axis=1)  # type: ignore
    df = df[keep].copy().reset_index(drop=True)

    for _, row in df.iterrows():
        if not row.new_source_tif.exists():
            Path(row.new_source_tif).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(row.source_tif, row.new_source_tif)

    df = df.drop(columns="source_tif").rename(columns={"new_source_tif": "source_tif"})

    df.to_csv(f"/Users/kyledorman/data/estuary/dataset/{out_name}.csv", index=False)

    return df


def run():
    LOW_QUALITY_LABELS = Path("/Volumes/x10pro/estuary/low_quality/labeled_data.csv")
    TIME_SERIES_LABELS = Path("/Volumes/x10pro/estuary/ca_all/dove/labels.csv")
    TRAIN_LABELS = Path("/Volumes/x10pro/estuary/dove/labels.csv")
    CLUSTER_LABELS = Path("/Volumes/x10pro/estuary/low_quality/cluster_labels.csv")
    VALID_REGIONS = [
        int(p.stem) for p in Path("/Volumes/x10pro/estuary/ca_grids").glob("*.geojson")
    ]

    lq_df = pd.read_csv(LOW_QUALITY_LABELS)
    ts_df = pd.read_csv(TIME_SERIES_LABELS)
    train_df = pd.read_csv(TRAIN_LABELS)
    clustered_df = pd.read_csv(CLUSTER_LABELS)

    for df in [lq_df, ts_df, train_df, clustered_df]:
        df["dove"] = df.source_tif.apply(lambda p: Path(p).parents[5].name)
        df["year"] = df.source_tif.apply(lambda p: int(Path(p).parents[3].name))
        df["month"] = df.source_tif.apply(lambda p: int(Path(p).parents[2].name))
        df["region"] = df.source_tif.apply(lambda p: int(Path(p).parents[1].name))
        df["asset_id"] = df.source_tif.apply(lambda p: Path(p).name.split("_3B_")[0])

    lq_df = lq_df.sort_values(by=["region", "asset_id"]).reset_index(drop=True)
    ts_df = ts_df.sort_values(by=["region", "asset_id"]).reset_index(drop=True)
    train_df = train_df.sort_values(by=["region", "asset_id"]).reset_index(drop=True)
    clustered_df = clustered_df.sort_values(by=["region", "asset_id"]).reset_index(drop=True)

    time_series_regions = ts_df.region.unique()
    print("Time Series Regions")
    print(time_series_regions.tolist())

    # Copy low quality labels for time series regions to time series dataset.
    # Perfer the label from time series to the label in low_quality
    ts_df["dup_order"] = 0
    lq_df["dup_order"] = 1
    ts_df = (
        pd.concat([ts_df, lq_df[lq_df.region.isin(time_series_regions)]])
        .sort_values(by=["region", "asset_id", "dup_order"])
        .reset_index(drop=True)
        .drop_duplicates(["region", "asset_id"])
        .drop(columns=["dup_order"])
        .reset_index(drop=True)
    )
    ts_df["source"] = "time_series"

    lq_df["dup_order"] = 0
    lq_df["source"] = "low_quality"
    clustered_df["source"] = "clustered"
    train_df["source"] = "train"
    train_df["dup_order"] = 1

    base_model_dataset = pd.concat([train_df, lq_df], ignore_index=True)
    # Only keep valid regions or unsure data
    base_model_dataset = base_model_dataset[
        (base_model_dataset.region.isin(VALID_REGIONS)) | (base_model_dataset.label == "unsure")
    ].copy()
    base_model_dataset = (
        base_model_dataset.sort_values(["region", "asset_id", "dup_order"])
        .drop_duplicates(["region", "asset_id"])
        .reset_index(drop=True)
        .drop(columns=["dup_order"])
    )

    process_df(ts_df, "time_series")
    process_df(base_model_dataset, "open_closed")
    process_df(clustered_df, "cluster_labels")


if __name__ == "__main__":
    run()
