from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

SKIP_SITE_IDS = {19, 29, 27, 13073, 54, 13027}


def _read_streamstats_points(gdb_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gdb_path, layer="GlobalWatershedPoint").to_crs(4326)
    gdf = gdf.rename(columns={"Name": "site_id"}).copy()
    if "site_id" not in gdf.columns:
        raise click.ClickException("GlobalWatershedPoint is missing Name/site_id")
    gdf["site_id"] = pd.to_numeric(gdf["site_id"], errors="coerce")
    gdf = gdf.dropna(subset=["site_id"]).copy()
    gdf["site_id"] = gdf["site_id"].astype(int)
    gdf = gdf.loc[~gdf["site_id"].isin(SKIP_SITE_IDS)].copy()
    gdf = gdf.sort_values("site_id").drop_duplicates(subset="site_id")
    if gdf.empty:
        raise click.ClickException("No StreamStats points remain after applying skip list")
    return gdf


def _read_nwm_network(network_path: Path, feature_id_col: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(network_path)
    if gdf.crs is None:
        raise click.ClickException("NWM network file has no CRS")
    if feature_id_col not in gdf.columns:
        raise click.ClickException(
            f"NWM network file is missing feature id column: {feature_id_col}"
        )
    gdf = gdf.to_crs(4326).copy()
    gdf = gdf.dropna(subset=[feature_id_col]).copy()
    gdf = gdf.rename(columns={"geom": "geometry"}).set_geometry("geometry")
    return gdf


@click.command()
@click.option(
    "--streamstats-gdb",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to StreamStats geodatabase containing GlobalWatershedPoint.",
)
@click.option(
    "--nwm-network-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to NWM reach network vector file.",
)
@click.option(
    "--feature-id-col",
    default="id",
    show_default=True,
    help="Column in the NWM network containing the reach identifier.",
)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path (.csv, .parquet, .geojson, .gpkg, .shp).",
)
def main(
    streamstats_gdb: Path,
    nwm_network_path: Path,
    feature_id_col: str,
    out_path: Path,
) -> None:
    streamstats_pts = _read_streamstats_points(streamstats_gdb)
    click.echo(
        f"Using {len(streamstats_pts)} StreamStats points after skipping {len(SKIP_SITE_IDS)} flagged site_ids"
    )
    nwm = _read_nwm_network(nwm_network_path, feature_id_col=feature_id_col)

    search_crs = "EPSG:3310"
    pts_proj = streamstats_pts.to_crs(search_crs)
    nwm_proj = nwm.to_crs(search_crs)

    optional_cols = [
        c
        for c in ["order_", "streamorde", "stream_order", "tofeature", "gages", "comid"]
        if c in nwm_proj.columns
    ]
    keep_cols = [feature_id_col] + optional_cols + ["geometry"]

    joined = gpd.sjoin_nearest(
        pts_proj,
        nwm_proj[keep_cols],
        how="left",
        distance_col="distance_to_reach_m",
    )

    # In rare cases sjoin_nearest can return multiple equally near reaches for the
    # same outlet point. Keep a single deterministic match per site_id.
    joined = joined.sort_values(["site_id", "distance_to_reach_m", feature_id_col]).drop_duplicates(
        subset=["site_id"],
        keep="first",
    )

    if joined[feature_id_col].isna().any():
        raise click.ClickException(
            "Failed to assign an NWM reach to one or more StreamStats points"
        )
    if joined["site_id"].duplicated().any():
        dupes = joined.loc[joined["site_id"].duplicated(keep=False), "site_id"].tolist()
        raise click.ClickException(
            f"Duplicate site_id assignments remain after deduplication: {dupes}"
        )

    joined = joined.to_crs(4326).copy()

    # Join the full reach geometry back in by matched id
    reach_geom = nwm[["id", "geometry"]].rename(columns={"geometry": "reach_geometry"})
    joined = joined.merge(reach_geom, on="id", how="left")

    # Turn it back into a GeoDataFrame with the point geometry active
    out = gpd.GeoDataFrame(joined, geometry="geometry", crs="EPSG:4326")

    out["point_lon"] = out.geometry.x
    out["point_lat"] = out.geometry.y

    cols = ["site_id", feature_id_col, "distance_to_reach_m", "point_lon", "point_lat"]
    cols += [c for c in optional_cols if c not in cols]
    cols += [c for c in ["Latitude", "Longitude", "WarningMsg", "HUCID"] if c in out.columns]
    cols += ["geometry"]

    out = out[cols].sort_values("site_id").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".csv":
        out.drop(columns="geometry").to_csv(out_path, index=False)
    elif suffix == ".parquet":
        out.drop(columns="geometry").to_parquet(out_path, index=False)
    elif suffix == ".geojson":
        out.to_file(out_path, driver="GeoJSON")
    elif suffix == ".gpkg":
        out.to_file(out_path, driver="GPKG")
    elif suffix == ".shp":
        out.to_file(out_path)
    else:
        raise click.ClickException(
            "Unsupported output format. Use .csv, .parquet, .geojson, .gpkg, or .shp"
        )

    click.echo(f"Wrote {len(out)} unique site-to-NWM reach matches to {out_path}")


if __name__ == "__main__":
    main()
