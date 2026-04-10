#!/usr/bin/env python3
import time
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import pandas as pd
import requests
import tqdm
from pyproj import Geod

MONITORING_LOCATIONS_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items"
)
TIME_SERIES_METADATA_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata/items"
)
DISCHARGE_CODE = "00060"
REQUEST_TIMEOUT = 60
MAX_ITEMS_PER_PAGE = 10000
MAX_RETRIES = 6
BACKOFF_SECONDS = 2.0


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = None
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_s = BACKOFF_SECONDS * attempt
                if retry_after is not None:
                    try:
                        sleep_s = min(10.0, float(retry_after))
                    except ValueError:
                        sleep_s = min(10.0, BACKOFF_SECONDS * attempt)
                else:
                    sleep_s = min(10.0, BACKOFF_SECONDS * attempt)
                click.echo(
                    f"Rate limited on {url} (attempt {attempt}/{MAX_RETRIES}). Sleeping {sleep_s:.1f}s..."
                )
                last_exc = requests.HTTPError(
                    f"429 Too Many Requests for url: {response.url}", response=response
                )
                time.sleep(sleep_s)
                continue
            if 500 <= response.status_code < 600:
                sleep_s = min(10.0, BACKOFF_SECONDS * attempt)
                click.echo(
                    f"Server error {response.status_code} on {url} "
                    f"(attempt {attempt}/{MAX_RETRIES}). Sleeping {sleep_s:.1f}s..."
                )
                last_exc = requests.HTTPError(
                    f"{response.status_code} Server Error for url: {response.url}",
                    response=response,
                )
                time.sleep(sleep_s)
                continue
            response.raise_for_status()
            return response.json()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            body = ""
            if response is not None:
                try:
                    body = response.text[:500]
                except Exception:
                    body = ""
            if attempt == MAX_RETRIES:
                if body:
                    click.echo(f"Final error body from {url}: {body}")
                break
            sleep_s = min(10.0, BACKOFF_SECONDS * attempt)
            click.echo(
                f"Request failed for {url} (attempt {attempt}/{MAX_RETRIES}): {exc}. "
                f"Sleeping {sleep_s:.1f}s..."
            )
            if body:
                click.echo(f"Response body: {body}")
            time.sleep(sleep_s)
    if last_exc is None:
        raise RuntimeError(f"Request failed for {url} but no exception was captured")
    raise last_exc


def _monitoring_locations_bbox_query(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> pd.DataFrame:
    params = {
        "f": "json",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "limit": MAX_ITEMS_PER_PAGE,
    }
    payload = _get_json(MONITORING_LOCATIONS_URL, params=params)
    features = payload.get("features", [])
    if not features:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties", {})
        geom = feature.get("geometry")
        coords = None
        if geom and geom.get("type") == "Point":
            coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        rows.append(
            {
                "monitoring_location_id": feature.get("id"),
                "agency_code": props.get("agency_code"),
                "site_no": props.get("monitoring_location_number"),
                "station_nm": props.get("monitoring_location_name"),
                "site_tp_cd": props.get("site_type_code"),
                "dec_long_va": coords[0],
                "dec_lat_va": coords[1],
                "geometry": geom,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = (
        pd.DataFrame(rows).drop_duplicates(subset=["monitoring_location_id"]).reset_index(drop=True)
    )

    # Keep stream-like monitoring locations only. The OGC monitoring-locations
    # endpoint exposes site_type/site_type_code fields, but the query parameter
    # used previously was not valid and caused 400 responses.
    if "site_tp_cd" in out.columns:
        out = out.loc[
            out["site_tp_cd"].fillna("").astype(str).str.upper().isin({"ST", "ST-TS"})
        ].copy()

    return out.reset_index(drop=True)


def _time_series_metadata_bbox_query(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> pd.DataFrame:
    params = {
        "f": "json",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "parameter_code": DISCHARGE_CODE,
        "limit": MAX_ITEMS_PER_PAGE,
    }
    payload = _get_json(TIME_SERIES_METADATA_URL, params=params)
    features = payload.get("features", [])
    if not features:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties", {})
        rows.append(
            {
                "monitoring_location_id": props.get("monitoring_location_id"),
                "parameter_code": props.get("parameter_code"),
                "statistic_id": props.get("statistic_id"),
                "computation_identifier": props.get("computation_identifier"),
                "computation_period_identifier": props.get("computation_period_identifier"),
                "begin": props.get("begin") or props.get("begin_utc"),
                "end": props.get("end") or props.get("end_utc"),
            }
        )

    out = pd.DataFrame(rows)
    if "monitoring_location_id" in out.columns:
        out = out.dropna(subset=["monitoring_location_id"]).reset_index(drop=True)
    return out


def _points_from_site_df(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    out = df.copy()
    out["dec_lat_va"] = out["dec_lat_va"].astype(float)
    out["dec_long_va"] = out["dec_long_va"].astype(float)
    gdf = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["dec_long_va"], out["dec_lat_va"]),
        crs="EPSG:4326",
    )
    return gdf


def _overall_bbox(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _expanded_watershed_search_polygons(
    watersheds: gpd.GeoDataFrame,
    outlets: gpd.GeoDataFrame,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Expand each watershed by unioning it with a circular buffer around the outlet point."""
    if buffer_m <= 0:
        return watersheds.copy()

    watersheds_wgs84 = watersheds.to_crs(4326).copy()
    outlets_wgs84 = outlets.to_crs(4326).copy()

    search_crs = "EPSG:3310"
    watersheds_proj = watersheds_wgs84.to_crs(search_crs)
    outlets_proj = outlets_wgs84.to_crs(search_crs)

    outlet_buffers = outlets_proj[["site_id", "geometry"]].copy()
    outlet_buffers["geometry"] = outlet_buffers.geometry.buffer(buffer_m)

    merged = watersheds_proj[["site_id", "geometry"]].merge(
        outlet_buffers[["site_id", "geometry"]].rename(columns={"geometry": "buffer_geom"}),
        on="site_id",
        how="left",
    )

    merged["geometry"] = merged.apply(
        lambda row: (
            row.geometry.union(row.buffer_geom)
            if row.buffer_geom is not None and pd.notna(row.buffer_geom)
            else row.geometry
        ),
        axis=1,
    )

    out = gpd.GeoDataFrame(
        merged[["site_id", "geometry"]],
        geometry="geometry",
        crs=search_crs,
    ).to_crs(4326)

    return out


def _candidate_gauges_for_all_watersheds(
    search_polygons: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = _overall_bbox(search_polygons)
    site_df = _monitoring_locations_bbox_query(minx, miny, maxx, maxy)
    if site_df.empty:
        return gpd.GeoDataFrame(site_df, geometry=[], crs="EPSG:4326")

    site_gdf = _points_from_site_df(site_df)
    if site_gdf.empty:
        return site_gdf

    join_cols = ["site_id", "geometry"]
    joined = gpd.sjoin(
        site_gdf,
        search_polygons[join_cols],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return gpd.GeoDataFrame(joined, geometry=[], crs="EPSG:4326")

    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    return joined.reset_index(drop=True)


def _series_catalog_summary(df: pd.DataFrame, start: str, end: str) -> dict[str, Any] | None:
    if df.empty:
        return None

    work = df.copy()
    if "parameter_code" in work.columns:
        work = work.loc[work["parameter_code"] == DISCHARGE_CODE].copy()
    if work.empty:
        return None

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    begin_ts = pd.to_datetime(work.get("begin"), errors="coerce")  # type: ignore
    end_ts_series = pd.to_datetime(work.get("end"), errors="coerce")  # type: ignore
    overlaps = (
        begin_ts.notna()
        & end_ts_series.notna()
        & (begin_ts <= end_ts)
        & (end_ts_series >= start_ts)
    )
    work = work.loc[overlaps].copy()
    if work.empty:
        return None

    is_dv = work["computation_period_identifier"].astype(str).str.lower() == "daily"
    is_iv = ~is_dv

    iv = work.loc[is_iv]
    dv = work.loc[is_dv]

    has_iv = not iv.empty
    has_dv = not dv.empty
    if not has_iv and not has_dv:
        return None

    data_types: list[str] = []
    if has_iv:
        data_types.append("iv")
    if has_dv:
        data_types.append("dv")

    return {
        "data_types_found": ",".join(data_types),
        "has_iv": has_iv,
        "iv_first_time": iv["begin"].min() if has_iv else None,
        "iv_last_time": iv["end"].max() if has_iv else None,
        "has_dv": has_dv,
        "dv_first_date": dv["begin"].min() if has_dv else None,
        "dv_last_date": dv["end"].max() if has_dv else None,
    }


@click.command()
@click.option("--watersheds-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--start", default="2018-01-01", show_default=True)
@click.option("--end", default="2024-12-31", show_default=True)
@click.option("--outlet-buffer-m", default=1000.0, show_default=True, type=float)
@click.option("--out-path", type=click.Path(path_type=Path), required=True)
def main(
    watersheds_path: Path,
    start: str,
    end: str,
    outlet_buffer_m: float,
    out_path: Path,
) -> None:
    watersheds = gpd.read_file(watersheds_path, layer="GlobalWatershed").to_crs(4326)
    outlets = gpd.read_file(watersheds_path, layer="GlobalWatershedPoint").to_crs(4326)

    watersheds = watersheds.rename(columns={"Name": "site_id"})
    outlets = outlets.rename(columns={"Name": "site_id"})

    if "site_id" not in watersheds.columns:
        raise click.ClickException("Missing site_id in watershed polygons")
    if "site_id" not in outlets.columns:
        raise click.ClickException("Missing site_id in watershed points")

    search_polygons = _expanded_watershed_search_polygons(
        watersheds=watersheds,
        outlets=outlets,
        buffer_m=outlet_buffer_m,
    )

    click.echo("Querying USGS monitoring locations once for the full watershed extent")
    candidate_sites = _candidate_gauges_for_all_watersheds(search_polygons)
    if candidate_sites.empty:
        raise click.ClickException("No candidate gauges found inside any watershed")
    click.echo(
        f"Found {len(candidate_sites)} site-in-watershed matches across "
        f"{candidate_sites['monitoring_location_id'].nunique()} unique gauges"
    )
    key = "monitoring_location_id"
    candidate_sites[key] = candidate_sites[key].astype(str)

    geod = Geod(ellps="WGS84")
    rows: list[dict[str, Any]] = []

    minx, miny, maxx, maxy = _overall_bbox(search_polygons)
    click.echo("Querying USGS time-series metadata once for the full watershed extent")
    ts_meta_all = _time_series_metadata_bbox_query(minx, miny, maxx, maxy)
    if ts_meta_all.empty:
        raise click.ClickException(
            "No discharge time-series metadata found in the watershed extent"
        )
    click.echo(
        f"Fetched {len(ts_meta_all)} time-series metadata rows across "
        f"{ts_meta_all['monitoring_location_id'].nunique()} unique gauges"
    )

    for poly_row in tqdm.tqdm(
        watersheds.itertuples(index=False),
        total=len(watersheds),
        desc="Assigning gauges to watersheds",
    ):
        site_id = poly_row.site_id
        site_gdf = candidate_sites.loc[candidate_sites["site_id"] == site_id].copy()
        if site_gdf.empty:
            continue

        outlet_lon = None
        outlet_lat = None
        match = outlets.loc[outlets["site_id"] == site_id]
        if not match.empty:
            outlet_lon = match.geometry.iloc[0].x  # type: ignore
            outlet_lat = match.geometry.iloc[0].y  # type: ignore

        for gauge in site_gdf.itertuples(index=False):
            if gauge.monitoring_location_id is None:
                continue

            ts_meta_df = ts_meta_all.loc[
                ts_meta_all["monitoring_location_id"].astype(str)
                == str(gauge.monitoring_location_id)
            ].copy()
            summary = _series_catalog_summary(ts_meta_df, start=start, end=end)
            if summary is None:
                continue

            dist_m = None
            if outlet_lon is not None and outlet_lat is not None:
                _, _, dist_m = geod.inv(
                    outlet_lon,
                    outlet_lat,
                    gauge.dec_long_va,
                    gauge.dec_lat_va,
                )

            rows.append(
                {
                    "site_id": site_id,
                    "monitoring_location_id": gauge.monitoring_location_id,
                    "usgs_site_no": gauge.site_no,
                    "site_name": gauge.station_nm,
                    "site_type": getattr(gauge, "site_tp_cd", None),
                    "site_lat": gauge.dec_lat_va,
                    "site_lon": gauge.dec_long_va,
                    "inside_watershed": True,
                    "distance_to_outlet_m": dist_m,
                    **summary,
                    "geometry": gauge.geometry,
                }
            )

    if not rows:
        raise click.ClickException("No candidate discharge gauges found")

    out_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    out_gdf = out_gdf.sort_values(
        ["site_id", "distance_to_outlet_m", "usgs_site_no"], na_position="last"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".parquet":
        out_gdf.drop(columns="geometry").to_parquet(out_path, index=False)
    elif suffix == ".csv":
        out_gdf.drop(columns="geometry").to_csv(out_path, index=False)
    else:
        out_gdf.to_file(out_path)

    print(f"Wrote {len(out_gdf)} candidate gauge rows to {out_path}")


if __name__ == "__main__":
    main()
