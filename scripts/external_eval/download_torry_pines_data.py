from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import click
import pandas as pd
import requests
import tqdm

from estuary.util.data_parsing import _iter_month_windows

BASE_URL = "http://torreypines.trnerr.org/model/qryRawDataCSV_JSON.cfm"

# Los Penasquitos Lagoon
REGION_ID = 11


def _window_to_query_dates(win_start: datetime, win_end: datetime) -> tuple[str, str]:
    """Return (data_start, data_end) date strings for the trnerr endpoint.

    The endpoint accepts date-only strings and appears to treat `endDateTime` as an
    exclusive bound in the examples (e.g., 01/07/2026 to 01/08/2026 gives one day).
    We therefore pass win_start.date() and win_end.date().
    """
    s = win_start.astimezone(UTC).strftime("%m/%d/%Y")
    e = win_end.astimezone(UTC).strftime("%m/%d/%Y")
    return s, e


def _fetch_window(
    session: requests.Session,
    win_start: datetime,
    win_end: datetime,
) -> list[tuple[datetime, float]] | None:
    data_start, data_end = _window_to_query_dates(win_start, win_end)

    url = f"{BASE_URL}?&dataSource=LPLNW&beginDateTime={data_start}&endDateTime={data_end}"
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    try:
        js = resp.json()
    except requests.JSONDecodeError:
        return None

    if not isinstance(js, list):
        raise ValueError(f"Unexpected flot response type: {type(js)}")

    data: list[tuple[datetime, float]] = []
    for item in js:
        if not isinstance(item, dict):
            continue

        dt_raw = item.get("Date")
        wl_raw = item.get("Water Level")
        if not dt_raw or wl_raw is None:
            continue

        # Example: "2026-01-01 00:30:00" (assumed UTC)
        try:
            ts_dt = datetime.strptime(str(dt_raw).strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        except ValueError:
            # If the format changes, skip rather than silently mis-parse.
            continue

        try:
            height_ft = float(str(wl_raw).strip())
            height_m = height_ft * 0.3048
        except ValueError:
            continue

        data.append((ts_dt, height_m))

    return data


def _append_rows(
    path: Path,
    rows: list[tuple[datetime, float]],
) -> None:
    if not rows:
        return

    records = []
    for ts_dt, height_m in rows:
        records.append(
            {
                "timestamp_utc": ts_dt,
                "height": height_m,
                "region": REGION_ID,
                "source": "trnerr",
                "sensor_id": "LPLNW",
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
    help="Start time (UTC). Example: 2017-01-01T00:00:00Z",
)
@click.option(
    "--end",
    type=click.DateTime(),
    required=True,
    help="End time (UTC). Example: 2026-01-01T00:00:00Z",
)
def main(save_dir: Path, start: datetime, end: datetime) -> None:
    user_start = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    user_end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    if user_end <= user_start:
        raise click.ClickException("--end must be after --start")

    with requests.Session() as session:
        out_csv = save_dir / str(REGION_ID) / "flot.csv"
        if out_csv.exists():
            os.remove(out_csv)

        to_run = list(_iter_month_windows(user_start, user_end))
        for win_start, win_end in tqdm.tqdm(to_run):
            try:
                data = _fetch_window(
                    session=session,
                    win_start=win_start,
                    win_end=win_end,
                )
            except requests.exceptions.RequestException as e:
                raise click.ClickException(f"""trnerr request failed: {e}""") from e

            if data is None or not len(data):
                continue

            _append_rows(
                path=out_csv,
                rows=data,
            )


if __name__ == "__main__":
    main()
