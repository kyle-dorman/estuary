from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click
import pandas as pd
import requests
import tqdm

# Hard-coded estuary -> USGS gauge mapping.
# A site_id may map to multiple gauges whose flows can later be combined.
SITE_GAUGES: dict[int, list[str]] = {
    96: ["11482500"],
    12103: ["11481200"],
    92: ["11469000"],
    84: ["11468000"],
    77: ["11467553", "11467510"],
    72: ["11467000", "11467200"],
    13057: ["11162570"],
    2138: ["11162500"],
    56: ["11161000"],
    57: ["11160000"],
    51: ["11159500"],
    50: ["11152500"],
    48: ["11143250"],
    33: ["11140585"],
    32: ["11136100"],
    31: ["11134000"],
    28: ["11120000", "11119940", "11120500", "11120520"],
    27: ["11119770", "11119750", "11119745"],
    22: ["11119500"],
    21: ["11118500"],
    20: ["11109000", "11113000", "11113500"],
    17: ["11046325", "11046360", "11046300"],
    16: ["11046100"],
    15: ["11046000"],
    14: ["11042000"],
    11: ["11023340"],
}

IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
BACKOFF_SECONDS = 2.0
DEFAULT_START = "2017-01-01"
DEFAULT_END = "2025-12-31"


def _fetch_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = None
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                sleep_s = min(10.0, BACKOFF_SECONDS * attempt)
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        sleep_s = min(10.0, float(retry_after))
                    except ValueError:
                        pass
                time.sleep(sleep_s)
                last_exc = requests.HTTPError(
                    f"429 Too Many Requests for url: {response.url}", response=response
                )
                continue
            if 500 <= response.status_code < 600:
                sleep_s = min(10.0, BACKOFF_SECONDS * attempt)
                time.sleep(sleep_s)
                last_exc = requests.HTTPError(
                    f"{response.status_code} Server Error for url: {response.url}",
                    response=response,
                )
                continue
            response.raise_for_status()
            return response.json()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            sleep_s = min(10.0, BACKOFF_SECONDS * attempt)
            time.sleep(sleep_s)
    if last_exc is None:
        raise RuntimeError(f"Request failed for {url} but no exception was captured")
    raise last_exc


def _series_rows(
    payload: dict[str, Any],
    site_id: int,
    gauge_id: str,
    data_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    time_series = payload.get("value", {}).get("timeSeries", [])

    for series in time_series:
        source_info = series.get("sourceInfo", {})
        variable = series.get("variable", {})
        values_blocks = series.get("values", [])

        variable_code = None
        variable_name = variable.get("variableName")
        unit = variable.get("unit", {}).get("unitCode")
        unit_name = variable.get("unit", {}).get("unitName")
        no_data_value = variable.get("noDataValue")
        options = variable.get("options", {})
        statistic_code = options.get("option", [{}])[0].get("value") if options else None

        variable_codes = variable.get("variableCode", [])
        if variable_codes:
            variable_code = variable_codes[0].get("value")

        site_name = source_info.get("siteName")
        site_code = gauge_id
        site_codes = source_info.get("siteCode", [])
        if site_codes:
            site_code = site_codes[0].get("value", gauge_id)
        network = site_codes[0].get("network") if site_codes else None

        geog = source_info.get("geoLocation", {}).get("geogLocation", {})
        lat = geog.get("latitude")
        lon = geog.get("longitude")

        site_props = source_info.get("siteProperty", [])
        huc_cd = None
        for prop in site_props:
            if prop.get("name") == "hucCd":
                huc_cd = prop.get("value")
                break

        for block in values_blocks:
            method = block.get("method", [])
            method_desc = method[0].get("methodDescription") if method else None
            qualifiers_meta = block.get("qualifier", [])
            qualifier_codes_available = (
                ",".join(
                    [q.get("qualifierCode", "") for q in qualifiers_meta if q.get("qualifierCode")]
                )
                or None
            )

            for obs in block.get("value", []):
                qualifiers = obs.get("qualifiers", [])
                rows.append(
                    {
                        "site_id": site_id,
                        "gauge_id": gauge_id,
                        "site_code": site_code,
                        "site_name": site_name,
                        "network": network,
                        "huc_cd": huc_cd,
                        "data_type": data_type,
                        "date_time": obs.get("dateTime"),
                        "value": obs.get("value"),
                        "qualifier_codes": ",".join(qualifiers) if qualifiers else None,
                        "qualifier_codes_available": qualifier_codes_available,
                        "variable_code": variable_code,
                        "variable_name": variable_name,
                        "statistic_code": statistic_code,
                        "unit": unit,
                        "unit_name": unit_name,
                        "no_data_value": no_data_value,
                        "method_description": method_desc,
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
    return rows


def fetch_site_data(
    site_id: int, gauge_id: str, start: str, end: str, data_type: str
) -> list[dict[str, Any]]:
    url = IV_URL if data_type == "iv" else DV_URL
    params = {
        "format": "json",
        "sites": gauge_id,
        "startDT": start,
        "endDT": end,
        "siteStatus": "all",
    }
    payload = _fetch_json(url, params)
    return _series_rows(payload, site_id, gauge_id, data_type)


@click.command()
@click.option("--start", default=DEFAULT_START, show_default=True)
@click.option("--end", default=DEFAULT_END, show_default=True)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to output parquet file.",
)
def main(start: str, end: str, out_path: Path) -> None:
    all_rows: list[dict[str, Any]] = []

    site_gauge_pairs = [
        (site_id, gauge_id) for site_id, gauge_ids in SITE_GAUGES.items() for gauge_id in gauge_ids
    ]

    for site_id, gauge_id in tqdm.tqdm(site_gauge_pairs, desc="Downloading USGS gauges"):
        for data_type in ("iv", "dv"):
            rows = fetch_site_data(
                site_id=site_id, gauge_id=gauge_id, start=start, end=end, data_type=data_type
            )
            all_rows.extend(rows)
            time.sleep(0.05)

    if not all_rows:
        raise click.ClickException("No data were downloaded for the requested date range")

    df = pd.DataFrame(all_rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Split IV and DV before datetime parsing (avoids mixed timezone issues)
    df_iv = df[df["data_type"] == "iv"].copy()
    df_dv = df[df["data_type"] == "dv"].copy()

    # -----------------------------
    # IV handling (true timestamps)
    # -----------------------------
    if not df_iv.empty:
        df_iv["date_time"] = pd.to_datetime(df_iv["date_time"], errors="coerce", utc=True)
        df_iv = df_iv.sort_values(
            ["site_id", "gauge_id", "variable_code", "date_time"]
        ).reset_index(drop=True)

    # -----------------------------
    # DV handling (daily values)
    # -----------------------------
    if not df_dv.empty:
        # parse but drop timezone (keep local calendar day)
        dt = pd.to_datetime(df_dv["date_time"], errors="coerce")
        df_dv["date_time_local"] = dt.dt.tz_localize(None)
        df_dv["date"] = df_dv["date_time_local"].dt.date

        df_dv = df_dv.sort_values(
            ["site_id", "gauge_id", "variable_code", "date_time_local"]
        ).reset_index(drop=True)

    # -----------------------------
    # Save outputs
    # -----------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iv_path = out_path.with_name(out_path.stem + "_iv" + out_path.suffix)
    dv_path = out_path.with_name(out_path.stem + "_dv" + out_path.suffix)

    if not df_iv.empty:
        df_iv.to_parquet(iv_path, index=False)
        click.echo(f"Wrote {len(df_iv)} IV rows to {iv_path}")
    else:
        click.echo("No IV data to write")

    if not df_dv.empty:
        df_dv.to_parquet(dv_path, index=False)
        click.echo(f"Wrote {len(df_dv)} DV rows to {dv_path}")
    else:
        click.echo("No DV data to write")


if __name__ == "__main__":
    main()
