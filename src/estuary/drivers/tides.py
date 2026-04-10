import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyTMD.compute import tide_elevations

logger = logging.getLogger(__name__)


def _coerce_time_array(ts: np.ndarray) -> np.ndarray:
    """Coerce input times to a numpy datetime64 array in UTC-compatible form."""
    ts_arr = np.asarray(ts)

    if np.issubdtype(ts_arr.dtype, np.datetime64):
        return ts_arr.astype("datetime64[ns]")

    if ts_arr.dtype == object:
        return pd.to_datetime(ts_arr).to_numpy(dtype="datetime64[ns]")

    raise TypeError(
        "ts must be a numpy datetime64 array or an object array convertible with pandas.to_datetime"
    )


def _validate_point_arrays(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs_arr = np.asarray(xs, dtype=float).reshape(-1)
    ys_arr = np.asarray(ys, dtype=float).reshape(-1)

    if xs_arr.shape != ys_arr.shape:
        raise ValueError(
            f"xs and ys must have the same shape, got {xs_arr.shape} and {ys_arr.shape}"
        )

    if not np.all(np.isfinite(xs_arr)):
        raise ValueError("xs contains non-finite values")
    if not np.all(np.isfinite(ys_arr)):
        raise ValueError("ys contains non-finite values")

    return xs_arr, ys_arr


def calc_tide_elevatons(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    model_directory: Path | None,
    definition_file: Path | None = None,
    model_name: str | None = None,
    crop_buffer_deg: float = 0.1,
) -> np.ndarray:
    """
    Compute tidal elevations for multiple spatial points over a shared time series
    using a pyTMD tidal model.

    Parameters
    ----------
    xs : np.ndarray
        Array of x-coordinates (typically longitudes) of shape (N,).
        Coordinate reference system must match the model after transformation.
    ys : np.ndarray
        Array of y-coordinates (typically latitudes) of shape (N,).
    ts : np.ndarray
        Array of datetime objects or timestamps representing the time series
        at which to compute tidal elevations. Shape (T,).
    model_directory : Path | None
        Path to the directory containing the pyTMD tidal model files.
        If None, pyTMD will attempt to use default model paths.
    definition_file : Path | None, default None
        Path to a pyTMD definition file describing a custom tidal model.
    model_name : str | None, default None
        Name of a tidal model available in the pyTMD model database.
        At least one of ``definition_file`` or ``model_name`` must be provided.
    crop_buffer_deg : float, default 0.1
        Buffer in decimal degrees used to crop the tidal model around each point.
        This reduces I/O and memory usage when using large models such as FES2022.

    Returns
    -------
    np.ndarray
        Array of tidal elevations with shape (N, T), where:
        - N is the number of spatial points
        - T is the length of the time series
    """
    xs_arr, ys_arr = _validate_point_arrays(xs, ys)
    ts_arr = _coerce_time_array(ts)

    if ts_arr.ndim != 1:
        raise ValueError(f"ts must be one-dimensional, got shape {ts_arr.shape}")
    if len(ts_arr) == 0:
        raise ValueError("ts is empty")

    if definition_file is None and model_name is None:
        raise ValueError("At least one of definition_file or model_name must be provided")

    model_label = definition_file.name if definition_file is not None else model_name
    logger.info(
        "Computing tides for %d point(s) and %d timestamp(s) using model=%s",
        len(xs_arr),
        len(ts_arr),
        model_label,
    )

    out: list[np.ndarray] = []

    # Loop over spatial points; each iteration computes a full time series
    # of tidal elevations for one (x, y) location.
    for i, (x, y) in enumerate(zip(xs_arr, ys_arr, strict=True)):
        logger.info("Computing tides for point %d/%d at lon=%s lat=%s", i + 1, len(xs_arr), x, y)

        elev = tide_elevations(
            x,
            y,
            delta_time=ts_arr,
            directory=model_directory,
            buffer=crop_buffer_deg,
            type="time series",
            standard="datetime",
            method="nearest",
            definition_file=definition_file,
            model=model_name,
        )
        elev_np = np.asarray(elev)

        if elev_np.ndim == 0:
            elev_np = elev_np.reshape(1)

        out.append(elev_np)

    return np.stack(out, axis=0)
