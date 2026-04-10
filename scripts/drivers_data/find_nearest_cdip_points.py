#!/usr/bin/env python3

import hashlib
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr
from pyproj import Geod

CATALOG_CONNECT_TIMEOUT = 20
CATALOG_READ_TIMEOUT = 180
CATALOG_MAX_RETRIES = 5
CATALOG_BACKOFF_SECONDS = 2.0


def _fetch_bytes_with_retries(url: str, timeout: tuple[int, int]) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, CATALOG_MAX_RETRIES + 1):
        try:
            click.echo(f"Fetching catalog XML (attempt {attempt}/{CATALOG_MAX_RETRIES}): {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            last_exc = exc
            if attempt == CATALOG_MAX_RETRIES:
                break
            sleep_s = CATALOG_BACKOFF_SECONDS * attempt
            click.echo(f"Catalog fetch failed ({exc}). Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _to_360(lon: np.ndarray | float) -> np.ndarray | float:
    return (lon + 360) % 360


def _extract_candidate_points(geometry) -> list[tuple[str, float, float]]:
    """
    Return candidate sampling points from a site geometry as
    (candidate_name, lon, lat) tuples in EPSG:4326.

    Uses centroid plus the four bounding-box corners. This is intended
    for the small square AOIs used to represent estuary mouths.
    """
    minx, miny, maxx, maxy = geometry.bounds
    centroid = geometry.centroid
    return [
        ("centroid", float(centroid.x), float(centroid.y)),
        ("ll", float(minx), float(miny)),
        ("ul", float(minx), float(maxy)),
        ("lr", float(maxx), float(miny)),
        ("ur", float(maxx), float(maxy)),
    ]


def _normalize_station_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode().strip()
    return str(value).strip()


def _thredds_catalog_xml_url(catalog_url: str) -> str:
    if catalog_url.endswith("catalog.html"):
        return catalog_url[: -len("catalog.html")] + "catalog.xml"
    if catalog_url.endswith("catalog.xml"):
        return catalog_url
    if catalog_url.endswith("/"):
        return catalog_url + "catalog.xml"
    return catalog_url + "/catalog.xml"


def _opendap_url_from_catalog_url_path(catalog_url: str, url_path: str) -> str:
    """Build a THREDDS OPeNDAP dataset URL from a catalog URL and dataset urlPath."""
    xml_url = _thredds_catalog_xml_url(catalog_url)
    if "/thredds/" not in xml_url:
        raise click.ClickException(f"Unexpected THREDDS catalog URL: {catalog_url}")
    server_root = xml_url.split("/thredds/", 1)[0]
    return f"{server_root}/thredds/dodsC/{url_path.lstrip('/')}"


def _dap2_url(url: str) -> str:
    if url.startswith("http://"):
        start = len("http://")
        return "dap2://" + url[start:]
    if url.startswith("https://"):
        start = len("https://")
        return "dap2://" + url[start:]
    return url


def _catalog_cache_path(cdip_meta_path: Path, catalog_url: str) -> Path:
    digest = hashlib.md5(catalog_url.encode()).hexdigest()[:12]
    return cdip_meta_path.with_name(f"{cdip_meta_path.stem}_catalog_{digest}.xml")


def _fetch_catalog_dataset_urls(catalog_url: str, cache_path: Path | None = None) -> list[str]:
    xml_url = _thredds_catalog_xml_url(catalog_url)
    if cache_path is not None and cache_path.exists():
        click.echo(f"Loading cached catalog XML from {cache_path}")
        xml_bytes = cache_path.read_bytes()
    else:
        click.echo(f"Getting root xml from {xml_url}")
        try:
            xml_bytes = _fetch_bytes_with_retries(
                xml_url,
                timeout=(CATALOG_CONNECT_TIMEOUT, CATALOG_READ_TIMEOUT),
            )
        except Exception as exc:
            if cache_path is not None and cache_path.exists():
                click.echo(
                    f"Catalog fetch failed ({exc}). Falling back to cached XML at {cache_path}"
                )
                xml_bytes = cache_path.read_bytes()
            else:
                raise
        else:
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(xml_bytes)
                click.echo(f"Cached catalog XML to {cache_path}")

    root = ET.fromstring(xml_bytes)
    ns = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}
    all_datasets = root.findall(".//t:dataset", ns)
    click.echo(f"Catalog contains {len(all_datasets)} dataset entries")

    service_map: dict[str, tuple[str, str]] = {}
    for service in all_datasets:
        name = service.attrib.get("name")
        service_type = service.attrib.get("serviceType", "")
        base = service.attrib.get("base", "")
        if name:
            service_map[name] = (service_type, base)

    dataset_urls: list[str] = []
    n_total = 0
    n_nc = 0
    n_hindcast = 0

    for dataset in tqdm.tqdm(all_datasets, desc="Scanning THREDDS catalog"):
        n_total += 1
        url_path = dataset.attrib.get("urlPath")

        if not url_path or not url_path.endswith(".nc"):
            continue
        n_nc += 1

        if "_hindcast.nc" not in url_path:
            continue
        n_hindcast += 1

        service_name = dataset.attrib.get("serviceName")
        if service_name and service_name in service_map:
            service_type, _base = service_map[service_name]
            if service_type.lower() == "opendap":
                dataset_urls.append(_opendap_url_from_catalog_url_path(catalog_url, url_path))
                continue

        dataset_urls.append(_opendap_url_from_catalog_url_path(catalog_url, url_path))

    click.echo(f"Scanned {n_total} entries | {n_nc} NetCDF | {n_hindcast} hindcast")
    click.echo(f"Collected {len(dataset_urls)} candidate dataset URLs")
    if not dataset_urls:
        raise click.ClickException(f"No NetCDF datasets found in THREDDS catalog: {xml_url}")

    return sorted(set(dataset_urls))


def _station_key_from_dataset_url(dataset_url: str) -> str:
    stem = Path(dataset_url).stem
    return stem.split("_", 1)[0]


def _dedupe_dataset_urls_by_station(dataset_urls: list[str]) -> list[str]:
    """Keep one representative dataset URL per CDIP station/transect id."""
    keep: dict[str, str] = {}
    for dataset_url in sorted(dataset_urls):
        station_key = _station_key_from_dataset_url(dataset_url)
        if station_key not in keep:
            keep[station_key] = dataset_url
    return list(keep.values())


def _extract_station_metadata(dataset_url: str) -> dict[str, object] | None:
    # request only metadata variables to avoid loading large arrays
    query_vars = (
        "metaSiteLabel,metaLatitude,metaLongitude,metaWaterDepth,metaShoreNormal,metaGridMapping"
    )
    dataset_url_query = _dap2_url(f"{dataset_url}?{query_vars}")

    try:
        ds = xr.open_dataset(
            dataset_url_query,
            decode_times=False,
            cache=False,
            engine="pydap",
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        click.echo(f"Skipping dataset due to open failure: {dataset_url} ({exc})")
        return None

    try:
        lon_name = next(
            (name for name in ["metaLongitude", "longitude", "lon"] if name in ds.variables), None
        )
        lat_name = next(
            (name for name in ["metaLatitude", "latitude", "lat"] if name in ds.variables), None
        )
        id_name = next(
            (
                name
                for name in [
                    "metaSiteLabel",
                    "metaStationName",
                    "station_name",
                    "metaStationId",
                    "station_id",
                    "metaID",
                ]
                if name in ds.variables
            ),
            None,
        )

        if lon_name is None or lat_name is None:
            click.echo(f"Skipping dataset with missing lon/lat metadata: {dataset_url}")
            return None

        lon_val = float(np.asarray(ds[lon_name]).squeeze())
        lat_val = float(np.asarray(ds[lat_name]).squeeze())

        if id_name is not None:
            cdip_id = _normalize_station_id(np.asarray(ds[id_name]).squeeze().item())
        else:
            cdip_id = Path(dataset_url).stem

        depth_name = next((name for name in ["metaWaterDepth"] if name in ds.variables), None)
        shore_name = next((name for name in ["metaShoreNormal"] if name in ds.variables), None)

        depth_val = float(np.asarray(ds[depth_name]).squeeze()) if depth_name else None
        shore_val = float(np.asarray(ds[shore_name]).squeeze()) if shore_name else None

        return {
            "cdip_id": cdip_id,
            "pt_lon_360": float(_to_360(lon_val)),
            "pt_lon": lon_val,
            "pt_lat": lat_val,
            "water_depth": depth_val,
            "shore_normal": shore_val,
            "dataset_url": dataset_url,
        }
    finally:
        ds.close()


def _fetch_station_metadata_rows_threaded(
    dataset_urls: list[str], max_workers: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    if max_workers == 1:
        try:
            for dataset_url in tqdm.tqdm(dataset_urls, desc="Reading CDIP alongshore metadata"):
                row = _extract_station_metadata(dataset_url)
                if row is not None:
                    rows.append(row)
        except KeyboardInterrupt as err:
            click.echo("\nMetadata crawl interrupted by user. No cache file was written.")
            raise click.Abort() from err
        return rows

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_station_metadata, dataset_url): dataset_url
            for dataset_url in dataset_urls
        }
        try:
            with tqdm.tqdm(total=len(futures), desc="Reading CDIP alongshore metadata") as pbar:
                for future in as_completed(futures):
                    row = future.result()
                    if row is not None:
                        rows.append(row)
                    pbar.update(1)
        except KeyboardInterrupt as err:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            click.echo("\nMetadata crawl interrupted by user. No cache file was written.")
            raise click.Abort() from err
    return rows


def _load_or_create_cdip_metadata(
    cdip_meta_path: Path, catalog_url: str, max_workers: int
) -> pd.DataFrame:
    if cdip_meta_path.exists():
        click.echo(f"Loading CDIP metadata from {cdip_meta_path}")
        if cdip_meta_path.suffix == ".parquet":
            meta_df = pd.read_parquet(cdip_meta_path)
        else:
            meta_df = pd.read_csv(cdip_meta_path)
    else:
        click.echo("CDIP metadata not found, creating from remote THREDDS catalog...")
        catalog_cache_path = _catalog_cache_path(cdip_meta_path, catalog_url)
        click.echo(f"Catalog cache path: {catalog_cache_path}")
        dataset_urls = _fetch_catalog_dataset_urls(
            catalog_url=catalog_url,
            cache_path=catalog_cache_path,
        )
        dataset_urls = _dedupe_dataset_urls_by_station(dataset_urls)
        click.echo(
            f"Found {len(dataset_urls)} unique CDIP alongshore datasets after station deduplication"
        )
        if max_workers < 1:
            raise click.ClickException("max-workers must be >= 1")
        rows = _fetch_station_metadata_rows_threaded(
            dataset_urls=dataset_urls,
            max_workers=max_workers,
        )

        if not rows:
            raise click.ClickException("Failed to build CDIP metadata from THREDDS catalog")

        meta_df = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["cdip_id"])
            .sort_values("cdip_id")
            .reset_index(drop=True)
        )
        cdip_meta_path.parent.mkdir(parents=True, exist_ok=True)
        if cdip_meta_path.suffix == ".parquet":
            meta_df.to_parquet(cdip_meta_path, index=False)
        else:
            meta_df.to_csv(cdip_meta_path, index=False)
        click.echo(f"Saved CDIP metadata to {cdip_meta_path}")

    required_columns = {"cdip_id", "pt_lon_360", "pt_lat"}
    missing = required_columns.difference(meta_df.columns)
    if missing:
        raise click.ClickException(
            f"CDIP metadata file is missing required columns: {sorted(missing)}"
        )

    return meta_df


CDIP_ALONGSHORE_CATALOG_URL = (
    "https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/MOP_alongshore/catalog.html"
)


@click.command()
@click.option(
    "--cdip-meta-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to local CDIP alongshore metadata file. Will be created if missing.",
)
@click.option(
    "--catalog-url",
    default=CDIP_ALONGSHORE_CATALOG_URL,
    show_default=True,
    help="THREDDS catalog URL for CDIP alongshore model datasets.",
)
@click.option(
    "--max-workers",
    default=4,
    show_default=True,
    type=int,
    help="Maximum number of concurrent metadata requests.",
)
@click.option(
    "--grid-points-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help=(
        "Path to a directory of per-site geospatial files readable by GeoPandas. "
        "Each filename stem is used as the site identifier."
    ),
)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output file. Recommended: .parquet, .csv, or .geojson",
)
@click.option(
    "--buffer-deg",
    default=1.0,
    show_default=True,
    type=float,
    help="Latitude/longitude buffer, in degrees, for subsetting candidate CDIP grid cells.",
)
def main(
    cdip_meta_path: Path,
    catalog_url: str,
    max_workers: int,
    grid_points_dir: Path,
    out_path: Path,
    buffer_deg: float,
) -> None:
    """
    Find the nearest valid CDIP model point for each site geometry in a directory.

    Output columns:
    - site_id
    - mouth_lon_360, mouth_lon, mouth_lat
    - pt_lon_360, pt_lon, pt_lat
    - cdip_id, dataset_url
    - dist_m, dist_km
    """
    # Load or create CDIP alongshore metadata
    try:
        meta_df = _load_or_create_cdip_metadata(
            cdip_meta_path=cdip_meta_path,
            catalog_url=catalog_url,
            max_workers=max_workers,
        )
    except KeyboardInterrupt as err:
        click.echo("\nInterrupted while loading or creating CDIP metadata.")
        raise click.Abort() from err

    valid_lon = meta_df["pt_lon_360"].to_numpy()
    valid_lat = meta_df["pt_lat"].to_numpy()
    valid_idx = np.arange(len(meta_df))

    click.echo(f"Reading site geometries from: {grid_points_dir}")

    site_files = sorted(
        [p for p in grid_points_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    )
    if not site_files:
        raise click.ClickException(f"No site files found in: {grid_points_dir}")

    site_candidates: list[dict[str, object]] = []
    for site_file in site_files:
        site_id = site_file.stem
        gdf = gpd.read_file(site_file)

        if gdf.empty:
            click.echo(f"Skipping empty site file: {site_file.name}")
            continue

        if gdf.crs is None:
            click.echo(f"Warning: {site_file.name} has no CRS defined. Assuming EPSG:4326.")
            gdf = gdf.set_crs(4326)
        elif str(gdf.crs).lower() != "epsg:4326":
            click.echo(f"Reprojecting {site_file.name} from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(4326)

        geom = gdf.geometry.iloc[0]
        for candidate_name, lon_raw, lat_raw in _extract_candidate_points(geom):
            site_candidates.append(
                {
                    "site_id": site_id,
                    "candidate_name": candidate_name,
                    "candidate_lon": lon_raw,
                    "candidate_lat": lat_raw,
                    "candidate_lon_360": _to_360(lon_raw),
                }
            )

    if not site_candidates:
        raise click.ClickException("No valid site geometries were found.")

    candidates_df = pd.DataFrame(site_candidates)

    min_lon = _to_360(candidates_df["candidate_lon"].min() - buffer_deg)
    max_lon = _to_360(candidates_df["candidate_lon"].max() + buffer_deg)
    min_lat = candidates_df["candidate_lat"].min() - buffer_deg
    max_lat = candidates_df["candidate_lat"].max() + buffer_deg

    # This simple filter assumes the longitude window does not wrap around 0.
    # That is true for California in [0, 360] space.
    mask = (
        (valid_lon > min_lon)
        & (valid_lon < max_lon)
        & (valid_lat > min_lat)
        & (valid_lat < max_lat)
    )
    filt_valid_lon = valid_lon[mask]
    filt_valid_lat = valid_lat[mask]
    filt_valid_idx = valid_idx[mask]

    if len(filt_valid_lon) == 0:
        raise click.ClickException("No valid CDIP points found inside the search window.")

    geod = Geod(ellps="WGS84")

    def nearest_valid_model_point(
        target_lon_360: float,
        target_lat: float,
        search_lon_360: np.ndarray,
        search_lat: np.ndarray,
        search_idx: np.ndarray,
    ) -> tuple[float, float, float, int]:
        _, _, dist_m = geod.inv(
            np.full(search_lon_360.shape, target_lon_360),
            np.full(search_lat.shape, target_lat),
            search_lon_360,
            search_lat,
        )
        i = int(np.argmin(dist_m))
        return (
            float(search_lon_360[i]),
            float(search_lat[i]),
            float(dist_m[i]),
            int(search_idx[i]),
        )

    candidate_rows: list[dict[str, object]] = []
    candidate_iter = candidates_df.itertuples(index=False)
    for row in tqdm.tqdm(
        candidate_iter,
        total=len(candidates_df),
        desc="Finding nearest valid cdip points",
    ):
        pt_lon_360, pt_lat, dist_m, idx = nearest_valid_model_point(
            target_lon_360=row.candidate_lon_360,  # pyright: ignore[reportArgumentType]
            target_lat=row.candidate_lat,  # pyright: ignore[reportArgumentType]
            search_lon_360=filt_valid_lon,
            search_lat=filt_valid_lat,
            search_idx=filt_valid_idx,
        )
        pt_lon = ((pt_lon_360 + 180) % 360) - 180

        candidate_rows.append(
            {
                "site_id": row.site_id,
                "candidate_name": row.candidate_name,
                "mouth_lon_360": row.candidate_lon_360,
                "mouth_lon": row.candidate_lon,
                "mouth_lat": row.candidate_lat,
                "pt_lon_360": pt_lon_360,
                "pt_lon": pt_lon,
                "pt_lat": pt_lat,
                "dist_m": dist_m,
                "dist_km": dist_m / 1000.0,
                "cdip_id": meta_df.iloc[idx]["cdip_id"],
                "dataset_url": meta_df.iloc[idx].get("dataset_url", None),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    out_df = (
        candidate_df.sort_values(["site_id", "dist_m", "candidate_name"])
        .groupby("site_id", as_index=False)
        .first()
        .sort_values("site_id")
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    click.echo(f"Writing output to: {out_path}")

    if suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        out_df.to_csv(out_path, index=False)
    elif suffix in {".geojson", ".gpkg", ".shp"}:
        out_gdf = gpd.GeoDataFrame(
            out_df,
            geometry=gpd.points_from_xy(out_df["pt_lon"], out_df["pt_lat"]),
            crs="EPSG:4326",
        )
        driver = None
        if suffix == ".geojson":
            driver = "GeoJSON"
        elif suffix == ".gpkg":
            driver = "GPKG"
        out_gdf.to_file(out_path, driver=driver)
    else:
        raise click.ClickException(
            "Unsupported output format. Use .parquet, .csv, .geojson, .gpkg, or .shp"
        )

    click.echo(f"Selected nearest points for {len(out_df)} sites.")
    click.echo("Candidate counts by selected candidate:")
    click.echo(out_df["candidate_name"].value_counts().to_string())

    click.echo("\nSummary:")
    click.echo(out_df["dist_km"].describe().to_string())


if __name__ == "__main__":
    main()
