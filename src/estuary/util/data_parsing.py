from collections.abc import Iterable
from datetime import UTC, datetime


def _dt_to_ms(dt: datetime) -> int:
    dt_utc = dt.astimezone(UTC)
    return int(dt_utc.timestamp() * 1000)


def _month_start(dt: datetime) -> datetime:
    dt = dt.astimezone(UTC)
    return datetime(dt.year, dt.month, 1, tzinfo=UTC)


def _add_one_month(dt: datetime) -> datetime:
    # dt must be UTC.
    year = dt.year
    month = dt.month
    if month == 12:
        return datetime(year + 1, 1, 1, tzinfo=UTC)
    return datetime(year, month + 1, 1, tzinfo=UTC)


def _iter_month_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    """Yield [month_start, month_end) windows covering [start, end)."""
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)

    cur = _month_start(start)
    while cur < end:
        nxt = _add_one_month(cur)
        win_start = max(cur, start)
        win_end = min(nxt, end)
        if win_start < win_end:
            yield win_start, win_end
        cur = nxt


def _convert_to_meters(values: list[tuple[int, float]], units: str) -> list[tuple[int, float]]:
    u = (units or "").strip().lower()
    if u in {"m", "meter", "meters"}:
        return values
    if u in {"ft", "foot", "feet"}:
        return [(ts, v * 0.3048) for ts, v in values]
    raise ValueError(f"Unsupported units from flot endpoint: {units!r}")
