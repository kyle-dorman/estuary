"""
Download USGS NWIS IV time series to partitioned Parquet via a Click CLI.

Example usage:

  python download_usgs.py \
    --sites 11467270 11162690 \
    --params 00010 00065 \
    --start 2017-01-01 --end 2024-12-31 \
    --outdir data/usgs_iv

Or using a file (one site id per line or a CSV with a 'site_no' column):

  python download_usgs.py \
    --sites-file sites.txt \
    --params 00010 00065 00060 \
    --start 2019-01-01 --end 2024-12-31 \
    --outdir data/usgs_iv
"""

import json
import pathlib
import time

import click
import numpy as np
import pandas as pd
import requests

# Default USGS IV endpoint (JSON). DV support would require a different parser.
IV_BASE_URL = "https://waterservices.usgs.gov/nwis/iv/"

# ---------------------------- helpers ---------------------------------------


def _read_sites_from_file(path: pathlib.Path) -> list[str]:
    """Load site IDs from a text/CSV file.
    - TXT: one site id per line
    - CSV: expects a column named 'site_no' (falls back to first column)
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".list", ".ids"}:
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    # try CSV/TSV via pandas
    df = pd.read_csv(path)
    col = "site_no" if "site_no" in df.columns else df.columns[0]
    return df[col].dropna().astype(str).unique().tolist()


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
    base_url: str, site: str, params: list[str], start: str, end: str
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
    r = requests.get(base_url, params=q, timeout=60)
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
            qualifiers = row.get("qualifiers") or []
            data.append(
                {
                    "value": value,
                    "qualifier": ",".join(qualifiers)
                    if isinstance(qualifiers, list)
                    else str(qualifiers),
                    "datetime": row.get("dateTime"),
                    "parameter_cd": variable_code,
                    "site_no": site,
                    "station_nm": site_name,
                }
            )

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["year_month"] = df["datetime"].dt.strftime("%Y-%m")
    df = (
        df.dropna(subset=["datetime"])  # drop bad timestamps
        .sort_values(["site_no", "parameter_cd", "datetime"])  # tidy order
        .reset_index(drop=True)
    )
    return df


def save_parquet_partitioned(df: pd.DataFrame, root: pathlib.Path) -> None:
    """Save partitioned by site_no and parameter_cd:
    {root}/SITE_NO/PARAM_CD/YYYY-MM.parquet
    """
    if df.empty:
        return
    for (site, param), sub in df.groupby(["site_no", "parameter_cd"], dropna=False):
        pdir = root / site / param
        pdir.mkdir(parents=True, exist_ok=True)
        sub = sub.copy()
        for ym, chunk in sub.groupby("year_month"):
            fpath = pdir / f"{ym}.parquet"
            assert not fpath.exists()
            chunk.drop(columns=["year_month"]).to_parquet(fpath, index=False, compression="snappy")


# ----------------------------- CLI ------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--sites",
    multiple=True,
    help="USGS site IDs (repeatable). Example: --sites 11467270 --sites 11162690",
)
@click.option(
    "--sites-file",
    type=click.Path(path_type=pathlib.Path),
    help="Path to a TXT (one id/line) or CSV (with 'site_no' column).",
)
@click.option(
    "--params",
    multiple=True,
    default=["00010", "00060", "00065", "63160"],
    show_default=True,
    help="USGS parameter codes (repeatable). E.g., 00010=temp, 00060=discharge, 00065=gage height, 63160=NAVD88 stream elevation",
)
@click.option("--start", required=True, help="Start date (YYYY-MM-DD).")
@click.option("--end", required=True, help="End date (YYYY-MM-DD).")
@click.option(
    "--outdir",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Output directory for partitioned Parquet.",
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
    sites: tuple[str, ...],
    sites_file: pathlib.Path | None,
    params: tuple[str, ...],
    start: str,
    end: str,
    outdir: pathlib.Path,
    days_per_chunk: int,
    requests_per_min: int,
    base_url: str,
):
    """Download USGS **Instantaneous Values (IV)** time series for selected sites/parameters.

    Note: This CLI currently supports the IV JSON endpoint. Daily Values (DV) support would
    require a slightly different parser and is not included in this refactor.
    """
    # Resolve sites
    sites_list: list[str] = list(sites)
    if sites_file is not None:
        sites_from_file = _read_sites_from_file(sites_file)
        sites_list.extend(sites_from_file)
    # de-dup & validate
    sites_list = [s.strip() for s in sites_list if str(s).strip()]
    sites_list = sorted(set(sites_list))
    if not sites_list:
        raise click.UsageError("No sites provided. Use --sites and/or --sites-file.")

    params_list = [p.strip() for p in params if str(p).strip()]
    if not params_list:
        raise click.UsageError("No parameters provided. Use --params.")

    outdir.mkdir(parents=True, exist_ok=True)
    chunks = daterange_chunks(start, end, days_per_chunk)

    req_i = 0
    for site in sites_list:
        frames: list[pd.DataFrame] = []
        for c_start, c_end in chunks:
            rate_limit(req_i, requests_per_min)
            req_i += 1
            try:
                df = fetch_nwis_iv_json(base_url, site, params_list, c_start, c_end)
                if not df.empty:
                    frames.append(df)
            except requests.HTTPError as e:
                click.echo(
                    f"[WARN] HTTP {e.response.status_code} for site={site} {c_start}..{c_end}: {e}"
                )
            except Exception as e:  # noqa: BLE001 — broad catch to keep the loop moving
                click.echo(f"[WARN] {e} for site={site} {c_start}..{c_end}")

        if frames:
            all_df = pd.concat(frames, ignore_index=True)
            all_df = all_df.drop_duplicates(
                subset=["site_no", "parameter_cd", "datetime"]
            )  # safety
            save_parquet_partitioned(all_df, outdir)

            # Write minimal station metadata once per site
            meta = all_df[["site_no", "station_nm"]].drop_duplicates()
            mdir = outdir / site
            mdir.mkdir(parents=True, exist_ok=True)
            (mdir / "site_metadata.json").write_text(meta.to_json(orient="records", indent=2))

            click.echo(f"[OK] Saved {len(all_df):,} rows for site {site}")
        else:
            click.echo(f"[INFO] No data for site {site} in {start}..{end}")


if __name__ == "__main__":
    main()
