"""
Download USGS NWIS IV time series to partitioned Parquet via a Click CLI.

Example usage:

  python download_usgs.py \
    --sites 11467270 11162690 \
    --start 2017-01-01 --end 2024-12-31 \
    --outdir data/usgs_iv

Or using a file (one site id per line or a CSV with a 'site_no' column):

  python download_usgs.py \
    --sites-file sites.txt \
    --start 2019-01-01 --end 2024-12-31 \
    --outdir data/usgs_iv
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import tqdm

# Default USGS IV endpoint (JSON). DV support would require a different parser.
IV_BASE_URL = "https://waterservices.usgs.gov/nwis/iv/"

# ---------------------------- helpers ---------------------------------------


def daterange_chunks(start: str, end: str, days_per_chunk: int) -> list[tuple[str, str]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    chunks: list[tuple[str, str]] = []
    cur = s
    while cur <= e:
        nxt = min(cur + pd.Timedelta(days=days_per_chunk - 1), e)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + pd.Timedelta(days=1)
    return chunks


def rate_limit(idx: int, per_min: int):
    """Basic pacing to avoid hammering the API."""
    delay = 60.0 / max(1, per_min)
    if idx > 0:
        time.sleep(delay)


def fetch_nwis_iv_json(
    base_url: str,
    site: str,
    params: list[str],
    start: str,
    end: str,
    region: int,
) -> pd.DataFrame:
    """Fetch NWIS Instantaneous Values (IV) as JSON and return a tidy DataFrame with:
    site_no, station_nm, parameter_cd, datetime (UTC), value (float), qualifier (str)
    """
    q = {
        "format": "json",
        "sites": site,
        "siteStatus": "all",
        "startDT": start,
        "endDT": end,
        "parameterCd": ",".join(params),
    }
    r = requests.get(base_url, params=q, timeout=10)
    r.raise_for_status()

    payload = json.loads(r.text)
    ts_list = payload.get("value", {}).get("timeSeries", [])
    if not ts_list:
        return pd.DataFrame()

    # Try to get station name from the first series; fall back gracefully
    try:
        site_name = ts_list[0]["sourceInfo"]["siteName"]
    except Exception:
        site_name = site

    data = []
    for var_data in ts_list:
        # Extract 5-digit parameter code robustly
        try:
            variable_code = str(
                var_data["variable"]["variableCode"][0]["value"]
            )  # typical JSON shape
        except Exception:
            # fallback if structure differs
            variable_code = str(var_data.get("variable", {}).get("value", ""))
        no_data = str(var_data.get("variable", {}).get("noDataValue", ""))

        values_blocks = var_data.get("values", [])
        if not values_blocks:
            continue
        for row in values_blocks[0].get("value", []):
            raw_val = row.get("value")
            value = float(raw_val) if (raw_val is not None and str(raw_val) != no_data) else np.nan
            data.append(
                {
                    "height": value,
                    "timestamp_utc": row.get("dateTime"),
                    "parameter_cd": variable_code,
                    "site_no": site,
                    "station_nm": site_name,
                    "region": region,
                    "source": "usgs",
                    "sensor_id": site,
                }
            )

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df = (
        df.dropna(subset=["timestamp_utc"])  # drop bad timestamps
        .sort_values(["site_no", "parameter_cd", "timestamp_utc"])  # tidy order
        .reset_index(drop=True)
    )

    return df


def _append_rows(
    path: Path,
    df: pd.DataFrame,
) -> None:
    if df.empty:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# ----------------------------- CLI ------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--sites-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to a vector file (GeoJSON/Shapefile/etc.) containing a 'usgs_site_no' column and a region id column.",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]),
    required=True,
    help="Start time (UTC). Common format: YYYY-MM-DD",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]),
    required=True,
    help="End time (UTC). Common format: YYYY-MM-DD",
)
@click.option(
    "--save-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output directory. Writes to save-dir/<region>/usgs.csv",
)
@click.option(
    "--days-per-chunk",
    type=int,
    default=90,
    show_default=True,
    help="Days per request chunk (smaller reduces payload size).",
)
@click.option(
    "--requests-per-min",
    type=int,
    default=30,
    show_default=True,
    help="Polite pacing for API requests.",
)
@click.option(
    "--base-url",
    default=IV_BASE_URL,
    show_default=True,
    help="Override NWIS IV base URL if needed.",
)
def main(
    sites_path: Path,
    start: datetime,
    end: datetime,
    save_dir: Path,
    days_per_chunk: int,
    requests_per_min: int,
    base_url: str,
):
    """Download USGS **Instantaneous Values (IV)** time series for selected sites/parameters.

    Note: This CLI currently supports the IV JSON endpoint. Daily Values (DV) support would
    require a slightly different parser and is not included in this refactor.
    """
    gdf = gpd.read_file(sites_path)

    gdf = gdf[~gdf.usgs_site_no.isna()]

    params_list = ["63160"]

    save_dir.mkdir(parents=True, exist_ok=True)

    start_utc = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    end_utc = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    if end_utc <= start_utc:
        raise click.ClickException("--end must be after --start")

    start_s = start_utc.strftime("%Y-%m-%d")
    end_s = end_utc.strftime("%Y-%m-%d")

    chunks = daterange_chunks(start_s, end_s, days_per_chunk)

    req_i = 0
    for _, row in tqdm.tqdm(gdf.iterrows(), total=len(gdf)):
        usgs_site_no = row["usgs_site_no"]
        region = str(row["Site code"])

        save_path = save_dir / region / "usgs.csv"
        if save_path.exists():
            os.remove(save_path)

        for c_start, c_end in chunks:
            req_i += 1
            try:
                df = fetch_nwis_iv_json(
                    base_url, usgs_site_no, params_list, c_start, c_end, row["Site code"]
                )
            except requests.HTTPError as e:
                click.echo(
                    f"[WARN] HTTP {e.response.status_code} for site={usgs_site_no} {c_start}..{c_end}: {e}"
                )
            except Exception as e:  # noqa: BLE001 — broad catch to keep the loop moving
                click.echo(f"[WARN] {e} for site={usgs_site_no} {c_start}..{c_end}")

            _append_rows(save_path, df)
            rate_limit(req_i, requests_per_min)


if __name__ == "__main__":
    main()
