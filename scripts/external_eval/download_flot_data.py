from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
import pandas as pd
import requests
import tqdm

from estuary.util.data_parsing import _convert_to_meters, _iter_month_windows

BASE_URL = "https://mcwrarealtimehydrodata.com"

# (site_id, device_id, region_id)
SITES: list[tuple[int, int, int]] = [
    (6208, 211, 50),  # Salinas River Lagoon
    (6193, 211, 48),  # Carmel River Lagoon
]


@dataclass(frozen=True)
class FlotSeries:
    units: str
    data: list[tuple[int, float]]  # (timestamp_ms, value)


def _window_to_query_dates(win_start: datetime, win_end: datetime) -> tuple[str, str]:
    """Return (data_start, data_end) date strings for the flot endpoint.

    The endpoint accepts date-only strings and appears to treat `data_end` as an
    exclusive bound in the examples (e.g., 2026-01-01 to 2026-01-02 gives one day).
    We therefore pass win_start.date() and win_end.date().
    """
    s = win_start.astimezone(UTC).date().isoformat()
    e = win_end.astimezone(UTC).date().isoformat()
    return s, e


def _fetch_window(
    session: requests.Session,
    site_id: int,
    device_id: int,
    win_start: datetime,
    win_end: datetime,
) -> FlotSeries | None:
    data_start, data_end = _window_to_query_dates(win_start, win_end)

    url = (
        f"{BASE_URL}/export/flot/?method=sensorDetails"
        f"&site_id={site_id}&device_id={device_id}"
        f"&data_start={data_start}&data_end={data_end}"
    )

    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    js = resp.json()

    if not isinstance(js, list):
        raise ValueError(f"Unexpected flot response type: {type(js)}")

    # Only one item in the list should have threshold == False.
    series_obj = None
    for obj in js:
        if isinstance(obj, dict) and obj.get("threshold") is False:
            series_obj = obj
            break

    if series_obj is None:
        raise RuntimeError("No response data")

    count = series_obj.get("count") or 0
    if count == 0:
        return None
    units = str(series_obj.get("units") or "")
    raw = series_obj.get("data") or []

    data: list[tuple[int, float]] = []
    for item in raw:
        # item looks like: [1767255081000, 8.45]
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            ts_ms = int(item[0])
            val = float(item[1])
        except (TypeError, ValueError):
            continue
        data.append((ts_ms, val))

    data_m = _convert_to_meters(data, units)
    return FlotSeries(units=units, data=data_m)


def _append_rows(
    path: Path,
    region: int,
    site_id: int,
    device_id: int,
    rows: list[tuple[int, float]],
) -> None:
    if not rows:
        return

    records = []
    for ts_ms, height_m in rows:
        ts_dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
        records.append(
            {
                "timestamp_utc": ts_dt,
                "height": height_m,
                "region": region,
                "source": "flot",
                "site_id": site_id,
                "device_id": device_id,
                "sensor_id": device_id,
            }
        )

    df = pd.DataFrame.from_records(records)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


@click.command()
@click.option(
    "--save-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory. Writes to save-dir/<region>/flot.csv",
)
@click.option(
    "--start",
    type=click.DateTime(),
    required=True,
    help="Start time (UTC). Example: 2025-01-01T00:00:00Z",
)
@click.option(
    "--end",
    type=click.DateTime(),
    required=True,
    help="End time (UTC). Example: 2026-01-01T00:00:00Z",
)
def main(save_dir: Path, start: datetime, end: datetime) -> None:
    """Download flot (MCWRA/SLO County Water) water level data 1 month at a time.

    - For each (site_id, device_id, region), queries month windows.
    - Picks the series where `threshold == False`.
    - Converts feet to meters when needed.
    - Skips (month, region) windows with no data.
    - Saves one CSV per region at: save-dir/<region>/flot.csv
    """

    user_start = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    user_end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    if user_end <= user_start:
        raise click.ClickException("--end must be after --start")

    # Stable ordering
    sites_sorted = sorted(SITES, key=lambda x: int(x[2]))

    with requests.Session() as session:
        for site_id, device_id, region in tqdm.tqdm(sites_sorted):
            out_csv = save_dir / str(region) / "flot.csv"
            if out_csv.exists():
                os.remove(out_csv)

            for win_start, win_end in _iter_month_windows(user_start, user_end):
                try:
                    series = _fetch_window(
                        session=session,
                        site_id=site_id,
                        device_id=device_id,
                        win_start=win_start,
                        win_end=win_end,
                    )
                except requests.exceptions.RequestException as e:
                    raise click.ClickException(
                        f"""flot request failed for region={region}, site_id={site_id},
                            device_id={device_id}: {e}"""
                    ) from e

                if series is None or not series.data:
                    continue

                _append_rows(
                    path=out_csv,
                    region=region,
                    site_id=site_id,
                    device_id=device_id,
                    rows=series.data,
                )


if __name__ == "__main__":
    main()
