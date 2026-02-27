import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd


def out_tif_path(row) -> Path:
    return (
        Path("/Users/kyledorman/data/estuary/dataset/images/")
        / row.instrument
        / "results"
        / str(row.year)
        / str(row.month)
        / str(row.region)
        / "files"
        / Path(row.source_tif).name
    )


def process_df(df: pd.DataFrame, out_name: str) -> pd.DataFrame:
    print(out_name)

    df["new_source_tif"] = df.apply(out_tif_path, axis=1)  # type: ignore

    keep = df.new_source_tif.apply(lambda a: Path(a).exists())
    if not keep.all():
        print("Removed missing files")
        print(df[~keep][["region", "year", "month", "instrument"]])

    df = df[keep].copy().reset_index(drop=True)

    for _, row in df.iterrows():
        if not row.new_source_tif.exists():
            Path(row.new_source_tif).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(row.source_tif, row.new_source_tif)

    df = df.drop(columns="source_tif").rename(columns={"new_source_tif": "source_tif"})

    df.to_csv(f"/Users/kyledorman/data/estuary/dataset/{out_name}.csv", index=False)

    return df


def dedupe_within_days(
    df: pd.DataFrame,
    days: int,
    keys: list[str],
    date_col: str,
) -> pd.DataFrame:
    """Deduplicate rows that occur within `days` of each other for the same key group.

    Keeps the earliest record, drops subsequent records within the window.

    Parameters
    ----------
    df : DataFrame
        Input labels.
    days : int
        Window size in days (inclusive).
    keys : list[str]
        Columns to group by (default: ['region', 'label']).
    date_col : str
        Name of the datetime column. If None, tries common options.

    Returns
    -------
    DataFrame
        Deduplicated DataFrame.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    def _dedupe_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col)
        keep_idx: list[int] = []
        last_kept = None
        for idx, d in zip(g.index.tolist(), g[date_col].tolist(), strict=True):
            if last_kept is None:
                keep_idx.append(idx)
                last_kept = d
                continue
            # drop if within window (<= days)
            if (d - last_kept).days > days:
                keep_idx.append(idx)
                last_kept = d
        return g.loc[keep_idx]

    out = (
        out.groupby(keys, group_keys=False, sort=False).apply(_dedupe_group).reset_index(drop=True)
    )

    return out


def run():
    LOW_QUALITY_LABELS = Path("/Volumes/x10pro/estuary/labeling/low_quality/unsure_only.csv")
    TIME_SERIES_LABELS = Path("/Volumes/x10pro/estuary/labeling/dove_timeseries/labels.csv")
    TRAIN_LABELS = Path("/Volumes/x10pro/estuary/labeling/dove_split_by_region/labels.csv")
    CLUSTER_LABELS = Path("/Volumes/x10pro/estuary/labeling/low_quality/cluster_labels.csv")
    gdf = gpd.read_file("/Volumes/x10pro/estuary/geos/ca_data_w_empa_pmep_usgs.geojson")
    VALID_REGIONS = gdf[~gdf.skipped]["Site code"].to_list()

    lq_df = pd.read_csv(LOW_QUALITY_LABELS)
    lq_df["acquired"] = pd.to_datetime(lq_df["acquired"], errors="coerce", utc=True)

    ts_df = pd.read_csv(TIME_SERIES_LABELS)
    ts_df["acquired"] = pd.to_datetime(ts_df["acquired"], errors="coerce", utc=True)

    train_df = pd.read_csv(TRAIN_LABELS)
    train_df["acquired"] = pd.to_datetime(train_df["acquired"], errors="coerce", utc=True)

    clustered_df = pd.read_csv(CLUSTER_LABELS)[
        ["acquired", "asset_id", "region", "source_tif", "unsure_label"]
    ]
    clustered_df["source_tif"] = clustered_df["source_tif"].str.replace("ca_all", "sat_data")
    clustered_df["acquired"] = pd.to_datetime(clustered_df["acquired"], errors="coerce", utc=True)

    for df in [lq_df, train_df, ts_df, clustered_df]:
        df["instrument"] = df.source_tif.apply(lambda p: Path(p).parents[5].name)
        df["year"] = df.source_tif.apply(lambda p: int(Path(p).parents[3].name))
        df["month"] = df.source_tif.apply(lambda p: int(Path(p).parents[2].name))
        df["region"] = df.source_tif.apply(lambda p: int(Path(p).parents[1].name))
        df["asset_id"] = df.source_tif.apply(lambda p: Path(p).name.split("_3B_")[0])

    lq_df = lq_df.sort_values(by=["region", "asset_id"]).reset_index(drop=True)
    ts_df = ts_df.sort_values(by=["region", "asset_id"]).reset_index(drop=True)
    train_df = train_df.sort_values(by=["region", "asset_id"]).reset_index(drop=True)
    clustered_df = clustered_df.sort_values(by=["region", "asset_id"]).reset_index(drop=True)

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

    time_series_regions = ts_df.region.unique()
    print("Time Series Regions")
    print(time_series_regions.tolist())

    # Copy train labels for time series regions to time series dataset.
    # Perfer the label from train to the label in time series
    ts_df["dup_order"] = 1
    ts_df["source"] = "time_series"
    base_model_dataset["dup_order"] = 0
    ts_df = (
        pd.concat([ts_df, base_model_dataset[base_model_dataset.region.isin(time_series_regions)]])
        .sort_values(by=["region", "asset_id", "dup_order"])
        .reset_index(drop=True)
        .drop_duplicates(["region", "asset_id"])
        .drop(columns=["dup_order"])
        .reset_index(drop=True)
    )
    base_model_dataset = base_model_dataset.drop(columns=["dup_order"])

    # Remove near-duplicate rows: same region + same label within 3 days.
    before = len(base_model_dataset)
    base_model_dataset = dedupe_within_days(
        base_model_dataset, days=3, keys=["region", "label"], date_col="acquired"
    )
    after = len(base_model_dataset)
    print(f"Deduped base_model_dataset within 3 days by region+label: {before} -> {after}")

    process_df(ts_df, "low_quality_time_series")
    process_df(base_model_dataset, "train")
    process_df(clustered_df, "cluster_labels")


if __name__ == "__main__":
    run()
