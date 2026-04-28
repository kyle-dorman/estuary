#!/usr/bin/env python3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import pandas as pd
import tqdm
import xarray as xr

WAVE_MAX_RETRIES = 4
WAVE_BACKOFF_SECONDS = 2.0


def _dap2_url(url: str) -> str:
    if url.startswith("http://"):
        start = len("http://")
        return "dap2://" + url[start:]
    if url.startswith("https://"):
        start = len("https://")
        return "dap2://" + url[start:]
    return url


def _wave_query_url(dataset_url: str) -> str:
    query_vars = "waveTime,waveHs,waveTp,waveDp,waveFlagPrimary,waveFlagSecondary"
    return _dap2_url(f"{dataset_url}?{query_vars}")


def _read_points(points_path: Path) -> pd.DataFrame:
    suffix = points_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(points_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(points_path)
    else:
        raise click.ClickException("points-path must be a .csv or .parquet file")

    required = {"site_id", "cdip_id", "dataset_url"}
    missing = required.difference(df.columns)
    if missing:
        raise click.ClickException(f"Points file is missing required columns: {sorted(missing)}")

    df = (
        df[list(required)]
        .drop_duplicates(subset=["site_id"])
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    return df


def _normalize_time_index(ds: xr.Dataset) -> pd.DatetimeIndex:
    if "waveTime" not in ds.variables and "waveTime" not in ds.coords:
        raise click.ClickException("Dataset is missing waveTime")

    time_index = pd.DatetimeIndex(pd.to_datetime(ds["waveTime"].values))
    if time_index.tz is not None:
        time_index = time_index.tz_convert(None)
    return time_index


def _series_or_none(ds: xr.Dataset, name: str, time_index: pd.DatetimeIndex) -> pd.Series:
    if name not in ds.variables:
        return pd.Series([pd.NA] * len(time_index), index=time_index)
    return pd.Series(ds[name].values, index=time_index)


def _extract_one_site(
    site_id: str,
    cdip_id: str,
    dataset_url: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    url = _wave_query_url(dataset_url)

    for attempt in range(1, WAVE_MAX_RETRIES + 1):
        try:
            ds = xr.open_dataset(
                url,
                decode_times=True,
                cache=False,
                engine="pydap",
            )
            break
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt == WAVE_MAX_RETRIES:
                raise
            sleep_s = WAVE_BACKOFF_SECONDS * attempt
            click.echo(
                f"[{site_id}] wave fetch failed on attempt {attempt}/{WAVE_MAX_RETRIES} "
                f"({exc}). Retrying in {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)

    try:
        time_index = _normalize_time_index(ds)
        mask = (time_index >= start) & (time_index <= end)
        if not mask.any():
            return None

        # good_waves = ( nc.variables['waveFlagPrimary'][:] == 1 )
        out = pd.DataFrame(
            {
                "time": time_index,
                "wave_hs": _series_or_none(ds, "waveHs", time_index).values,
                "wave_tp": _series_or_none(ds, "waveTp", time_index).values,
                "wave_dp": _series_or_none(ds, "waveDp", time_index).values,
                "wave_flag_primary": _series_or_none(ds, "waveFlagPrimary", time_index).values,
                "wave_flag_secondary": _series_or_none(ds, "waveFlagSecondary", time_index).values,
            }
        )
        out = out.loc[mask].copy()
        out.insert(0, "cdip_id", cdip_id)
        out.insert(0, "site_id", site_id)
        return out.reset_index(drop=True)
    finally:
        ds.close()


def _fetch_waves_threaded(
    points_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_workers: int,
) -> pd.DataFrame:
    if max_workers < 1:
        raise click.ClickException("max-workers must be >= 1")

    rows: list[pd.DataFrame] = []

    if max_workers == 1:
        iterator = points_df.itertuples(index=False)
        try:
            for row in tqdm.tqdm(iterator, total=len(points_df), desc="Reading CDIP wave data"):
                df = _extract_one_site(
                    site_id=row.site_id,  # type: ignore
                    cdip_id=row.cdip_id,  # type: ignore
                    dataset_url=row.dataset_url,  # type: ignore
                    start=start,
                    end=end,
                )
                if df is not None:
                    rows.append(df)
        except KeyboardInterrupt as err:
            click.echo("\nWave download interrupted by user. No output file was written.")
            raise click.Abort() from err
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _extract_one_site,
                    row.site_id,  # type: ignore
                    row.cdip_id,  # type: ignore
                    row.dataset_url,  # type: ignore
                    start,
                    end,
                ): row.site_id
                for row in points_df.itertuples(index=False)
            }
            try:
                with tqdm.tqdm(total=len(futures), desc="Reading CDIP wave data") as pbar:
                    for future in as_completed(futures):
                        df = future.result()
                        if df is not None:
                            rows.append(df)
                        pbar.update(1)
            except KeyboardInterrupt as err:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                click.echo("\nWave download interrupted by user. No output file was written.")
                raise click.Abort() from err

    if not rows:
        raise click.ClickException(
            "No wave data were downloaded for the requested sites/time range"
        )

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["site_id", "time"]).reset_index(drop=True)
    return out


@click.command()
@click.option(
    "--points-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to CSV or Parquet file containing nearest CDIP points.",
)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output parquet file.",
)
@click.option(
    "--start",
    default="2017-01-01",
    show_default=True,
    help="Start timestamp for wave extraction.",
)
@click.option(
    "--end",
    default="2025-12-31",
    show_default=True,
    help="End timestamp for wave extraction.",
)
@click.option(
    "--max-workers",
    default=4,
    show_default=True,
    type=int,
    help="Maximum number of concurrent CDIP dataset requests.",
)
def main(
    points_path: Path,
    out_path: Path,
    start: str,
    end: str,
    max_workers: int,
) -> None:
    """Download CDIP alongshore wave data for selected sites and save long-format parquet."""
    points_df = _read_points(points_path)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise click.ClickException("--end must be after --start")

    click.echo(f"Reading wave data for {len(points_df)} sites")
    click.echo(f"Time range: {start_ts} to {end_ts}")

    wave_df = _fetch_waves_threaded(
        points_df=points_df,
        start=start_ts,
        end=end_ts,
        max_workers=max_workers,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wave_df.to_parquet(out_path, index=False)

    click.echo(
        f"Wrote wave data for {wave_df['site_id'].nunique()} sites "
        f"({len(wave_df)} rows) to {out_path}"
    )


if __name__ == "__main__":
    main()
