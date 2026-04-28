import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from pyTMD import io as pio
from timescale import from_datetime

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
    model_directory: Path,
    definition_file: Path,
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
    model_directory : Paths.
        If None, pyTMD will attempt to use default model paths.
    definition_file : Path
        Path to a pyTMD definition file describing a custom tidal model.
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

    model_label = definition_file.name
    logger.info(
        "Computing tides for %d point(s) and %d timestamp(s) using model=%s",
        len(xs_arr),
        len(ts_arr),
        model_label,
    )

    out: list[np.ndarray] = []

    elev = tide_elevations(
        xs_arr,
        ys_arr,
        delta_time=ts_arr,
        directory=model_directory,
        buffer=crop_buffer_deg,
        # type="time series",
        # standard="datetime",
        # method="nearest",
        definition_file=definition_file,
    )

    out.append(elev)

    return np.stack(out, axis=0)


def tide_elevations(
    xs_arr: np.ndarray,
    ys_arr: np.ndarray,
    delta_time: np.ndarray,
    directory: Path,
    definition_file: Path,
    buffer: int | float,
) -> np.ndarray:
    """
    Compute ocean or load tides at points and times from
    model constituents

    Kyle: I took control of this fn so we could loop without reloading the model so many times. Sad.

    Parameters
    ----------
    x: np.ndarray
        x-coordinates
    y: np.ndarray
        y-coordinates
    delta_time: np.ndarray
        Time coordinates
    directory: str or NoneType, default None
        working data directory for tide models
    definition_file: str, pathlib.Path, io.IOBase or NoneType, default None
        Tide model definition file for use
    buffer: int or float, default 0
        Buffer distance for cropping tide model data

    Returns
    -------
    tpred: xarray.DataArray
        Predicted tide elevation (meters)
    """
    # default keyword arguments
    kwargs = {}
    kwargs.setdefault("chunks", None)
    kwargs.setdefault("corrections", None)
    kwargs.setdefault("append_node", False)

    method = "nearest"
    type = "time series"
    crs = 4326
    extrapolate: bool = False
    cutoff: int | float = 10.0

    # get parameters for tide model
    m = pio.model(directory).from_file(definition_file)

    # open dataset
    ds = m.open_dataset(group="z", chunks=kwargs["chunks"], append_node=kwargs["append_node"])

    # nodal corrections to apply
    nodal_corrections = m.corrections
    # minor constituents to infer
    minor_constituents = m.minor

    # convert delta times or datetimes objects to timescale
    ts = from_datetime(delta_time)

    # delta time (TT - UT1) for tide model
    if nodal_corrections in ("OTIS", "ATLAS", "TMD3", "netcdf"):
        # use delta time at 2000.0 to match TMDv2.5 outputs
        deltat = np.zeros_like(ts.tt_ut1)
    else:
        # use interpolated delta times
        deltat = ts.tt_ut1

    tpred_arr = []

    # Loop over spatial points; each iteration computes a full time series
    # of tidal elevations for one (x, y) location.
    for i, (x, y) in tqdm.tqdm(enumerate(zip(xs_arr, ys_arr, strict=True)), total=len(xs_arr)):
        logger.info("Computing tides for point %d/%d at lon=%s lat=%s", i + 1, len(xs_arr), x, y)

        # convert coordinates to xarray DataArrays
        # in coordinate reference system of model
        X, Y = ds.tmd.coords_as(x, y, type=type, crs=crs)

        # crop tide model dataset to buffer
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)
        # crop dataset to buffered default bounds
        ds_crop = ds.tmd.crop([xmin, xmax, ymin, ymax], buffer=buffer)

        # interpolate model to grid points
        local = ds_crop.tmd.interp(X, Y, method=method, extrapolate=extrapolate, cutoff=cutoff)
        # calculate tide values for input data type
        tpred = local.tmd.predict(ts.tide, deltat=deltat, corrections=nodal_corrections)
        # calculate values for minor constituents by inference
        # infer minor constituents
        tinfer = local.tmd.infer(
            ts.tide,
            deltat=deltat,
            corrections=nodal_corrections,
            minor=minor_constituents,
        )
        # add major and minor components
        tpred += tinfer
        # add attributes for inferred constituents
        tpred.attrs["inferred"] = []
        if hasattr(tinfer, "constituents"):
            tpred.attrs["inferred"].extend(tinfer.constituents)
        # add attributes
        tpred.attrs["nodal_corrections"] = nodal_corrections
        # return the ocean or load tide correction
        tpred_arr.append(np.array(tpred))

    out = np.array(tpred_arr)
    if out.ndim == 0:
        out = out.reshape(1)
    return out
