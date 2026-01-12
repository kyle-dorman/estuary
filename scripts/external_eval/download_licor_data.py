import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import pandas as pd
import requests
import tqdm

from estuary.util.data_parsing import _dt_to_ms, _iter_month_windows

# licor dataChannel, region_id
DATA_CHANNELS = [
    ("0942b00d-ca65-460d-9b34-8ca6f3912883", 25),  # Devereux Slough
    ("d7218efc-fea9-40f3-8789-ca39e78c0f1d", 51),  # Pajaro Lagoon
    ("f2ed8c17-707b-40ee-8002-1f44498a19c1", 2138),  # Pescadero Lagoon
    ("1908e69d-3637-41ff-a2b0-2a6fb679b8de", 13057),  # San Gregorio Creek
    ("d0cd3768-c0ea-45c8-b144-bb552fb67db6", 20),  # Santa Clara River
    ("cba5a28f-af65-4413-8a44-cc82963170ba", 21),  # Ventura River
    ("4aa09afb-a9b6-4327-a95f-d6c4bcaeeced", 53),  # Younger Lagoon
]

API_URL = "https://www.licor.cloud/api/dashboard/public/query"

# Dashboard/widget id captured from the public dashboard request
DASHBOARD_QUERY_ID = "cd5b1933-9022-44e4-a3c8-f4cfcc2a3433"


@dataclass(frozen=True)
class DataChannelInfo:
    data_channel: str
    region: int
    first_measurement: datetime | None
    last_measurement: datetime | None
    sensor_id: int | None


def _parse_licor_zulu(ts: str | None) -> datetime | None:
    """Parse LICOR Cloud Z timestamps like '2025-02-10T19:50:00Z' to UTC datetime."""
    if not ts:
        return None
    # Avoid dateutil dependency.
    # Expected format: YYYY-MM-DDTHH:MM:SSZ
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        return dt
    except ValueError:
        return None


def _build_payload(data_channel: str, start_ms: int, end_ms: int) -> dict:
    return {
        "id": DASHBOARD_QUERY_ID,
        "query": {
            "limit": 10000,
            "metrics": [
                {
                    "aggregators": [
                        {
                            "name": "avg",
                            "align_start_time": False,
                            "sampling": {"value": 10, "unit": "minutes"},
                        }
                    ],
                    "name": "com.onset.sensordata.waterlevel_si",
                    "exclude_tags": True,
                    "group_by": [],
                    "tags": {"dataChannel": [data_channel]},
                }
            ],
            "start_absolute": start_ms,
            "end_absolute": end_ms,
        },
    }


def _fetch_month(
    session: requests.Session,
    data_channel: str,
    region: int,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[DataChannelInfo, list[tuple[int, float]]]:
    """Fetch one month window. Returns (metadata, values) where values are (ts_ms, height_m)."""
    headers = {"content-type": "application/json"}
    payload = _build_payload(data_channel, _dt_to_ms(start_dt), _dt_to_ms(end_dt))

    resp = session.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    js = resp.json()

    q0 = (js.get("queries") or [{}])[0]

    # Metadata lives at response.json()["queries"][0]["dataChannel"]
    dc_meta = q0.get("dataChannel") or {}

    info = DataChannelInfo(
        data_channel=data_channel,
        region=region,
        first_measurement=_parse_licor_zulu(dc_meta.get("firstMeasurementTime")),
        last_measurement=_parse_licor_zulu(dc_meta.get("lastMeasurementTime")),
        sensor_id=dc_meta.get("sensor_key"),
    )

    results = q0.get("results") or []
    if not results:
        return info, []

    # Typical shape: results[0]["values"] = [[ts_ms, height], ...]
    values = (results[0] or {}).get("values") or []
    out: list[tuple[int, float]] = []
    for v in values:
        # v is like [1764884700000, 1.4842049407530102]
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            ts_ms = int(v[0])
            height = float(v[1])
        except (TypeError, ValueError):
            continue
        out.append((ts_ms, height))

    return info, out


def _append_rows(
    path: Path,
    region: int,
    data_channel: str,
    sensor_id: int | None,
    rows: list[tuple[int, float]],
) -> None:
    if not rows:
        return

    records = []
    for ts_ms, height in rows:
        ts_dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
        records.append(
            {
                "timestamp_utc": ts_dt,
                "height": height,
                "region": region,
                "source": "licor",
                "data_channel": data_channel,
                "sensor_id": sensor_id,
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
    help="Output directory. Writes to save-dir/<region>/licor.csv",
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
    """Download LICOR Cloud water level data 1 month at a time for each region.

    - Skips (month, region) windows with no data.
    - Saves one CSV per region at: save_dir/<region>/licor.csv
    """
    user_start = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    user_end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    if user_end <= user_start:
        raise click.ClickException("--end must be after --start")

    # Stable ordering for reproducibility
    channels_sorted = sorted(DATA_CHANNELS, key=lambda x: int(x[1]))

    with requests.Session() as session:
        for data_channel, region in tqdm.tqdm(channels_sorted):
            out_csv = save_dir / str(region) / "licor.csv"
            if out_csv.exists():
                os.remove(out_csv)

            # First request: get metadata (first/last measurement times) so we can constrain
            # queries. Use a small window near the user start to keep response sizes down.
            probe_start = user_start
            probe_end = min(user_end, user_start + timedelta(days=1))

            try:
                initial_info, _ = _fetch_month(
                    session, data_channel, region, probe_start, probe_end
                )
            except requests.exceptions.RequestException as e:
                raise click.ClickException(
                    f"LICOR request failed for region={region}, data_channel={data_channel}: {e}"
                ) from e

            # If LICOR doesn't report bounds, fall back to user bounds.
            avail_start = initial_info.first_measurement or user_start
            avail_end = initial_info.last_measurement or user_end

            effective_start = max(user_start, avail_start)
            effective_end = min(user_end, avail_end)

            if effective_end <= effective_start:
                # No overlap between user requested window and available data window
                continue

            for win_start, win_end in _iter_month_windows(effective_start, effective_end):
                info, rows = _fetch_month(session, data_channel, region, win_start, win_end)

                if not rows:
                    # No data for this (month, region)
                    continue

                _append_rows(out_csv, region, data_channel, info.sensor_id, rows)


if __name__ == "__main__":
    main()
