from pathlib import Path
from typing import Literal

import click
import fsspec
import pandas as pd
import xarray as xr

V21_ZARR_URL = "s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr"
V30_ZARR_URL = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/CONUS/zarr/chrtout.zarr"


def _read_matches(path: Path, feature_id_col: str) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".geojson", ".gpkg", ".shp"}:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        df = pd.DataFrame(gdf.drop(columns="geometry"))
    else:
        raise click.ClickException(
            "Unsupported matches format. Use .csv, .parquet, .geojson, .gpkg, or .shp"
        )

    required = {"site_id", feature_id_col}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(f"Matches file is missing required columns: {sorted(missing)}")

    df = df[["site_id", feature_id_col]].copy()
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce")
    df[feature_id_col] = pd.to_numeric(df[feature_id_col], errors="coerce")
    df = df.dropna(subset=["site_id", feature_id_col]).copy()
    df["site_id"] = df["site_id"].astype(int)
    df[feature_id_col] = df[feature_id_col].astype("int64")
    df = df.drop_duplicates(subset=["site_id"])
    return df


def _open_nwm_dataset(version: Literal["2.1", "3.0"]) -> xr.Dataset:
    if version == "2.1":
        mapper = fsspec.get_mapper(V21_ZARR_URL, anon=True)
    elif version == "3.0":
        mapper = fsspec.get_mapper(V30_ZARR_URL)
    else:
        raise click.ClickException("Unsupported NWM version. Use 2.1 or 3.0.")

    ds = xr.open_zarr(mapper, consolidated=True)

    required = {"feature_id", "time", "streamflow"}
    missing = required - set(ds.variables) - set(ds.coords)  # type: ignore
    if missing:
        raise click.ClickException(
            f"NWM dataset is missing expected variables/coords: {sorted(missing)}"
        )

    return ds


@click.command()
@click.option(
    "--matches-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Output from assign_nwm_reaches.py",
)
@click.option(
    "--feature-id-col",
    default="id",
    show_default=True,
    help="Feature/reach id column in the matches file.",
)
@click.option(
    "--start", required=True, type=str, help="YYYY-MM-DD", default="2017-01-01", show_default=True
)
@click.option(
    "--end", required=True, type=str, help="YYYY-MM-DD", default="2022-12-31", show_default=True
)
@click.option(
    "--nwm-version",
    type=click.Choice(["2.1", "3.0"]),
    default="3.0",
    show_default=True,
    help="Retrospective NWM version to query.",
)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output parquet path.",
)
def main(
    matches_path: Path,
    feature_id_col: str,
    start: str,
    end: str,
    nwm_version: str,
    out_path: Path,
) -> None:
    matches = _read_matches(matches_path, feature_id_col=feature_id_col)

    if nwm_version == "3.0":
        # Registry says v3.0 retrospective ends Jan 2023.
        end_ts = pd.Timestamp(end)
        if end_ts > pd.Timestamp("2023-01-31 23:00:00"):
            raise click.ClickException(
                "NWM v3.0 retrospective only runs through 2023-01-31. "
                "Trim --end or use a separate operational-data workflow for later dates."
            )

    if nwm_version == "2.1":
        end_ts = pd.Timestamp(end)
        if end_ts > pd.Timestamp("2020-12-31 23:00:00"):
            raise click.ClickException("NWM v2.1 retrospective only runs through 2020-12-31.")

    ds = _open_nwm_dataset(nwm_version)  # type: ignore # feature_id, time, streamflow

    feature_ids = matches[feature_id_col].astype("int64").tolist()

    subset = ds["streamflow"].sel(
        feature_id=feature_ids,
        time=slice(pd.Timestamp(start), pd.Timestamp(end)),
    )

    df = subset.to_dataframe().reset_index()

    if df.empty:
        raise click.ClickException(
            "No NWM rows returned for the requested time range and features."
        )

    df = df.merge(
        matches.rename(columns={feature_id_col: "feature_id"}),
        on="feature_id",
        how="left",
    )

    df = df.rename(columns={"streamflow": "streamflow_cms"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[["site_id", "feature_id", "time", "streamflow_cms"]].copy()
    df = df.sort_values(["site_id", "time"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    click.echo(
        f"Wrote {len(df)} rows for {df['site_id'].nunique()} sites "
        f"and {df['feature_id'].nunique()} NWM reaches to {out_path}"
    )


if __name__ == "__main__":
    main()
