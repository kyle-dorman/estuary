import json
import multiprocessing as mp
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import rasterio
from rasterio.env import Env
from tqdm import tqdm


# helper so we can use imap_unordered and keep tqdm progress
def _star_compute(args):
    return run_one(*args)


def _iter_task_grid(
    regions_root: Path,
    years: Iterable[int],
    months: Iterable[int],
    doves: Iterable[str],
) -> Iterable[tuple[str, int, int, str]]:
    """Generator that yields (region, year, month, dove) tuples.
    Avoids creating a giant list of tasks in memory.
    """
    for region_p in regions_root.iterdir():
        region = region_p.stem
        for year in years:
            for month in months:
                for dove in doves:
                    yield (region, year, month, dove)


def run_one(region: str, year: int, month: int, dove: str) -> list[dict[str, Any]]:
    """Collect per-band image stats for one (region, year, month, dove) cell.

    Fixes bugs in the original implementation:
    - Computes per-band means/stds over spatial dims (axis=(1,2)) instead of per-pixel means.
    - Opens the correct TIFF (tif_path) for spectral stats (previously opened the UDM twice).
    - Avoids reassigning `pth` accidentally; uses distinct variables.
    """
    out: list[dict[str, Any]] = []

    search_json_path = (
        Path(f"/Volumes/x10pro/estuary/ca_all/{dove}/results")
        / str(year)
        / str(month)
        / str(region)
        / "filtered_search_results.json"
    )
    if not search_json_path.exists():
        return out

    with open(search_json_path) as f:
        asset_data = json.load(f)

    # Consider both bases. If a file exists in multiple bases, we'll emit one row per base.
    for base in ["ca_all", "low_quality"]:
        files_dir = (
            Path(f"/Volumes/x10pro/estuary/{base}/{dove}/results")
            / str(year)
            / str(month)
            / str(region)
            / "files"
        )
        if not files_dir.exists():
            continue

        for d in asset_data:
            asset_id = d["id"]

            udm_path = next(files_dir.glob(f"{asset_id}*udm2_clip.tif"), None)
            if udm_path is None:
                continue

            if dove == "superdove":
                tif_path = next(files_dir.glob(f"{asset_id}*_SR_8b_clip.tif"), None)
            else:
                tif_path = next(files_dir.glob(f"{asset_id}*_SR_clip.tif"), None)
            if udm_path is None or tif_path is None:
                continue

            # Use Rasterio Env to allow internal threading; read as float32 to reduce memory.
            with Env(
                CPL_VSIL_CURL_CACHE_SIZE=0,  # local files; keep minimal overhead
                GDAL_CACHEMAX=256,  # MB cache
            ):
                # --- UDM2 stats (per-band mean over H,W) ---
                with rasterio.open(udm_path) as src_udm:
                    # Limit to available bands (expect up to 8 on UDM2; read first 7 as before)
                    udm_count = min(7, src_udm.count)
                    udm = src_udm.read(indexes=list(range(1, udm_count + 1)), out_dtype="float32")
                    # Means across spatial dims -> shape (udm_count,)
                    udm_means = udm.mean(axis=(1, 2))

                # --- Spectral TIFF stats (per-band mean/std over H,W) ---
                with rasterio.open(tif_path) as src_tif:
                    if src_tif.count == 4:
                        idx = [1, 2, 3, 4]
                    else:
                        # superdove 8-band: select 2,4,6,8 as in original
                        idx = [2, 4, 6, 8]
                    arr = src_tif.read(indexes=idx, out_dtype="float32")
                    tif_means = arr.mean(axis=(1, 2))
                    tif_stds = arr.std(axis=(1, 2))
                    import numpy as np

            # arr shape: (B, H, W) with B=4 (RGBNIR selection)
            vis = arr[0:3]  # B,G,R
            mean_vis_img = vis.mean(axis=0)  # per-pixel mean across visible bands
            gray = mean_vis_img.astype("float32")

            # Scene-level percentiles (visible)
            p90_vis = float(np.percentile(mean_vis_img, 90))
            p95_vis = float(np.percentile(mean_vis_img, 95))

            # Scene-adaptive threshold: bright if above median + 1.0 * std
            m = float(mean_vis_img.mean())
            s = float(mean_vis_img.std())
            bright_thresh = m + 1.0 * s
            bright_frac = float((mean_vis_img > bright_thresh).mean())

            # Low-texture proxy: per-pixel std across visible bands, then fraction below small
            # threshold
            std_vis_img = vis.std(axis=0)
            low_tex_frac = float((std_vis_img < (std_vis_img.mean() * 0.5)).mean())

            # NIR ratio (scene-level)
            nir_ratio = float(arr[3].mean() / (mean_vis_img.mean() + 1e-6))

            # Brightness & texture
            mean_brightness = float(gray.mean())
            std_texture = float(gray.std())
            lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

            # Inter-band correlation (smooth clouds -> high correlation)
            corr_bg = float(np.corrcoef(vis[0].ravel(), vis[1].ravel())[0, 1])
            corr_rg = float(np.corrcoef(vis[2].ravel(), vis[1].ravel())[0, 1])

            # Spectral consistency (low in noise)
            spec_consistency = (corr_bg + corr_rg) / 2

            props = dict(d.get("properties", {}))
            props["asset_id"] = asset_id
            props["year"] = year
            props["dove"] = dove
            props["month"] = month
            props["region"] = region
            props["base"] = base
            props["source_tif"] = str(tif_path)
            props["source_udm"] = str(udm_path)

            for i, val in enumerate(udm_means, start=1):
                props[f"udm_mean_{i}"] = float(val)
            for i, val in enumerate(tif_means, start=1):
                props[f"tif_mean_{i}"] = float(val)
            for i, val in enumerate(tif_stds, start=1):
                props[f"tif_std_{i}"] = float(val)

            props["p90_vis"] = p90_vis
            props["p95_vis"] = p95_vis
            props["bright_frac"] = bright_frac
            props["low_tex_frac"] = low_tex_frac
            props["nir_ratio"] = nir_ratio
            props["mean_brightness"] = mean_brightness
            props["std_texture"] = std_texture
            props["lap_var"] = lap_var
            props["spec_consistency"] = spec_consistency

            out.append(props)

    return out


def run():
    regions_root = Path("/Volumes/x10pro/estuary/ca_grids")
    years = range(2017, 2026)
    months = range(1, 13)
    doves = ("superdove", "dove")

    # Output directory and base file names
    out_dir = Path("/Volumes/x10pro/estuary/ca_all")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / "data_stats"
    final_parquet = out_base.with_suffix(".parquet")

    num_workers = max(1, min(mp.cpu_count() - 1, 16))
    chunksize = 32  # reduce scheduler overhead for many small tasks

    # Stream results with periodic flushes
    buffer: list[dict[str, Any]] = []
    chunk_idx = 0
    chunk_files = []

    with mp.Pool(processes=num_workers) as pool:
        tasks_iter = _iter_task_grid(regions_root, years, months, doves)
        total_tasks = sum(1 for _ in _iter_task_grid(regions_root, years, months, doves))
        for chunk_result in tqdm(
            pool.imap_unordered(_star_compute, tasks_iter, chunksize=chunksize),
            total=total_tasks,
            desc="Processing tasks",
        ):
            if not chunk_result:
                continue
            buffer.extend(chunk_result)

            # Flush in batches to keep memory bounded
            if len(buffer) >= 10000:
                df_chunk = pd.DataFrame(buffer)
                chunk_file = out_base.with_name(f"{out_base.name}_chunk_{chunk_idx}.parquet")
                df_chunk.to_parquet(chunk_file, index=False)
                chunk_files.append(chunk_file)
                chunk_idx += 1
                buffer.clear()

    # Final flush
    if buffer:
        df_chunk = pd.DataFrame(buffer)
        chunk_file = out_base.with_name(f"{out_base.name}_chunk_{chunk_idx}.parquet")
        df_chunk.to_parquet(chunk_file, index=False)
        chunk_files.append(chunk_file)
        buffer.clear()

    # Concatenate all chunk files into final parquet for convenience
    if chunk_files:
        dfs = [pd.read_parquet(f) for f in chunk_files]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_parquet(final_parquet, index=False)

        # Optionally, remove chunk files after concatenation
        for f in chunk_files:
            f.unlink()


if __name__ == "__main__":
    run()
