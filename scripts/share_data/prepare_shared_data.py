from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

DEFAULT_OUTPUT_DIR = Path("/Volumes/x10pro/estuary/shared_data")
DEFAULT_METADATA_PATH = Path("/Volumes/x10pro/estuary/geos/ca_data_w_empa_pmep_usgs.geojson")
DEFAULT_AOI_DIR = Path("/Volumes/x10pro/estuary/grids/valid")
DEFAULT_SKYSAT_LABELS_PATH = Path("/Volumes/x10pro/estuary/labeling/skysat/labels.csv")
DEFAULT_PLANETSCOPE_LABELS_PATH = Path("/Users/kyledorman/data/estuary/dataset/train.csv")
DEFAULT_PLANETSCOPE_WATER_LABELS_PATH = Path(
    "/Volumes/x10pro/estuary/labeling/dove_water/labels.csv"
)
DEFAULT_IMAGE_PREDS_PATH = Path(
    "/Users/kyledorman/data/results/estuary/train/20260305-095554/all_regions_timeseries_preds.csv"
)
DEFAULT_DAILY_PREDS_PATH = Path(
    "/Users/kyledorman/data/results/estuary/train/20260305-095554/"
    "merged_all_regions_timeseries_preds.csv"
)

LABEL_MAP = {
    "closed": "closed",
    "open": "open",
    "perched open": "partial",
    "unsure": "low_quality",
}

DATASET_TITLE = (
    "Data for: Seasonal and interannual dynamics of estuary connectivity along the California coast"
)
DATASET_VERSION = "1.0.0"
DATASET_DOI = "10.5281/zenodo.20753031"
DATASET_DOI_URL = f"https://doi.org/{DATASET_DOI}"
MANUSCRIPT_TITLE = (
    "Seasonal and interannual dynamics of estuary connectivity along the California coast"
)
CODE_REPOSITORY_URL = "https://github.com/kyle-dorman/estuary"
CONTACT_EMAIL = "kyle.e.dorman@gmail.com"
LICENSE_NAME = "Creative Commons Attribution 4.0 International"
LICENSE_SPDX = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def clean_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.eq(""))


def to_utc_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def format_utc(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def image_filename_from_path(path: Any) -> str:
    return Path(str(path)).name


def asset_id_from_image_path(path: Any) -> str:
    name = image_filename_from_path(path)
    for marker in ["_3B_", "_pansharpened"]:
        if marker in name:
            return name.split(marker, maxsplit=1)[0]
    return Path(name).stem


def image_id(site_id: pd.Series, asset_id: pd.Series) -> pd.Series:
    return site_id.astype("int64").astype(str) + "_" + asset_id.astype(str)


def parse_bool_flag(series: pd.Series, column_name: str) -> pd.Series:
    raw = series.astype("string").str.strip().str.lower()
    parsed = raw.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
            "yes": True,
            "no": False,
        }
    )

    if parsed.isna().any():
        examples = series[parsed.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"Unparseable boolean values in {column_name}: {examples}")

    return parsed.astype(bool)


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GeoJSON")


def clean_sites(metadata_path: Path) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    gdf = gpd.read_file(metadata_path)
    gdf = gdf[gdf["skipped"] == False].copy()  # noqa: E712

    site_name = clean_string(gdf["Site name"])
    site_name = site_name.fillna(clean_string(gdf["Estuary_Name"]))

    site_df = pd.DataFrame(
        {
            "site_id": gdf["Site code"].astype("int64"),
            "site_name": site_name,
            "pmep_estuary_id": gdf["PMEP_EstuaryID"].astype("Int64"),
            "pmep_estuary_name": clean_string(gdf["Estuary_Name"]),
            "pmep_data_source": clean_string(gdf["Data_Source"]),
            "cmecs_class": clean_string(gdf["CMECS_Class"]),
            "pmep_region": clean_string(gdf["PMEP_Region"]),
            "pmep_system_order": pd.to_numeric(gdf["System_Order"], errors="coerce"),
            "estuary_area_ha": pd.to_numeric(gdf["Estuary_Hectares"], errors="coerce"),
            "pmep_record_key": clean_string(gdf["Link"]),
            "empa_site_id": clean_string(gdf["siteid"]),
            "mouth_lat": pd.to_numeric(gdf["point_geom_lat"], errors="coerce"),
            "mouth_lon": pd.to_numeric(gdf["point_geom_long"], errors="coerce"),
            "pmep_point_lat": pd.to_numeric(gdf["pmep_pt_geom_lat"], errors="coerce"),
            "pmep_point_lon": pd.to_numeric(gdf["pmep_pt_geom_long"], errors="coerce"),
            "usgs_site_no": clean_string(gdf["usgs_site_no"]),
            "usgs_station_name": clean_string(gdf["usgs_station_nm"]),
            "usgs_site_lat": pd.to_numeric(gdf["usgs_site_latitude"], errors="coerce"),
            "usgs_site_lon": pd.to_numeric(gdf["usgs_site_longitude"], errors="coerce"),
        }
    )

    site_df["has_pmep_match"] = site_df["pmep_estuary_id"].notna()
    site_df["has_usgs_match"] = site_df["usgs_site_no"].notna()
    site_df = site_df.sort_values("site_id").reset_index(drop=True)

    geometry = [Point(xy) for xy in zip(site_df["mouth_lon"], site_df["mouth_lat"], strict=True)]
    sites_gdf = gpd.GeoDataFrame(site_df.copy(), geometry=geometry, crs="EPSG:4326")

    return site_df, sites_gdf


def clean_label_file(
    labels_path: Path,
    valid_site_ids: set[int],
    label_set: str,
) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(labels_path)
    df["site_id"] = df["region"].astype("int64")

    before = len(df)
    df = df[df["site_id"].isin(valid_site_ids)].copy()
    removed = before - len(df)

    acquired = to_utc_datetime(df["acquired"])
    raw_label = clean_string(df["label"])
    public_label = raw_label.map(LABEL_MAP)

    unknown_labels = sorted(raw_label[public_label.isna()].dropna().unique().tolist())
    if unknown_labels:
        raise ValueError(f"Unknown labels in {labels_path}: {unknown_labels}")

    out = pd.DataFrame(
        {
            "image_id": image_id(df["site_id"], df["asset_id"]),
            "site_id": df["site_id"],
            "date": acquired.dt.strftime("%Y-%m-%d"),
            "acquired_utc": format_utc(acquired),
            "year": acquired.dt.year.astype("Int64"),
            "month": acquired.dt.month.astype("Int64"),
            "instrument": clean_string(df["instrument"]),
            "asset_id": clean_string(df["asset_id"]),
            "image_filename": df["source_tif"].apply(image_filename_from_path),
            "label": public_label,
            "raw_label": raw_label,
            "label_set": label_set,
        }
    )

    if "source" in df.columns:
        out["sample_source"] = clean_string(df["source"])

    out = out.sort_values(["site_id", "acquired_utc", "image_id"]).reset_index(drop=True)
    return out, removed


def clean_image_predictions(
    predictions_path: Path,
    valid_site_ids: set[int],
) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    df["site_id"] = df["region"].astype("int64")
    df = df[df["site_id"].isin(valid_site_ids)].copy()

    acquired = to_utc_datetime(df["acquired"])
    p_open = pd.to_numeric(df["p_open"], errors="coerce")
    p_low_quality = pd.to_numeric(df["y_prob_unsure"], errors="coerce")
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
    low_quality_flag = parse_bool_flag(df["y_pred_unsure"], "y_pred_unsure")

    invalid = (~low_quality_flag) & y_pred.isna()
    if invalid.any():
        bad = df.loc[invalid, ["site_id", "asset_id"]].head(10).to_dict("records")
        raise ValueError(
            "Image-level predictions contain usable rows without y_pred values. "
            f"First examples: {bad}"
        )

    predicted_state = pd.Series("closed", index=df.index)
    predicted_state.loc[y_pred == 1] = "open"
    predicted_state.loc[low_quality_flag] = "low_quality"

    out = pd.DataFrame(
        {
            "image_id": image_id(df["site_id"], df["asset_id"]),
            "site_id": df["site_id"],
            "date": acquired.dt.strftime("%Y-%m-%d"),
            "acquired_utc": format_utc(acquired),
            "instrument": clean_string(df["instrument"]),
            "asset_id": clean_string(df["asset_id"]),
            "image_filename": df["source_tif"].apply(image_filename_from_path),
            "p_open": p_open,
            "p_low_quality": p_low_quality,
            "low_quality_flag": low_quality_flag.astype(bool),
            "predicted_state": predicted_state,
        }
    )

    return out.sort_values(["site_id", "acquired_utc", "image_id"]).reset_index(drop=True)


def clean_daily_predictions(
    predictions_path: Path,
    valid_site_ids: set[int],
) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    df["site_id"] = df["region"].astype("int64")
    df = df[df["site_id"].isin(valid_site_ids)].copy()

    acquired = to_utc_datetime(df["acquired"])
    p_open = pd.to_numeric(df["p_open"], errors="coerce")
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
    low_quality_flag = parse_bool_flag(df["y_pred_unsure"], "y_pred_unsure")

    invalid = (~low_quality_flag) & y_pred.isna()
    if invalid.any():
        bad = df.loc[invalid, ["site_id", "date"]].head(10).to_dict("records")
        raise ValueError(
            f"Daily predictions contain usable rows without y_pred values. First examples: {bad}"
        )

    predicted_state = pd.Series("closed", index=df.index)
    predicted_state.loc[y_pred == 1] = "open"
    predicted_state.loc[low_quality_flag] = "low_quality"

    out = pd.DataFrame(
        {
            "site_id": df["site_id"],
            "date": pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "acquired_utc": format_utc(acquired),
            "asset_id": df["source_tif"].apply(asset_id_from_image_path),
            "representative_image_filename": df["source_tif"].apply(image_filename_from_path),
            "p_open": p_open,
            "low_quality_flag": low_quality_flag,
            "predicted_state": predicted_state,
        }
    )

    return out.sort_values(["site_id", "date"]).reset_index(drop=True)


def merge_aoi_polygons(aoi_dir: Path, site_df: pd.DataFrame) -> gpd.GeoDataFrame:
    site_ids = set(site_df["site_id"].astype(int))
    pieces = []

    for path in sorted(aoi_dir.glob("*.geojson"), key=lambda p: int(p.stem)):
        site_id = int(path.stem)
        if site_id not in site_ids:
            continue

        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        gdf = gdf[["geometry"]].copy()
        gdf["site_id"] = site_id
        pieces.append(gdf)

    if not pieces:
        raise ValueError(f"No AOI polygons found in {aoi_dir}")

    aoi = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs="EPSG:4326")
    aoi = aoi.merge(
        site_df[
            [
                "site_id",
                "site_name",
                "pmep_estuary_id",
                "pmep_estuary_name",
                "cmecs_class",
                "pmep_region",
                "estuary_area_ha",
                "mouth_lat",
                "mouth_lon",
            ]
        ],
        on="site_id",
        how="left",
    )

    missing_aoi = sorted(site_ids - set(aoi["site_id"].astype(int)))
    if missing_aoi:
        raise ValueError(f"Missing AOI polygons for site_id values: {missing_aoi}")

    return aoi.sort_values("site_id").reset_index(drop=True)


def write_data_dictionary(path: Path) -> None:
    rows = [
        ("sites.csv", "site_id", "Stable numeric estuary/site identifier."),
        ("sites.csv", "site_name", "Common site name used in this study."),
        ("sites.csv", "pmep_estuary_id", "PMEP estuary identifier, when matched."),
        ("sites.csv", "pmep_estuary_name", "Estuary name from the PMEP inventory."),
        ("sites.csv", "cmecs_class", "CMECS estuary class from PMEP."),
        ("sites.csv", "pmep_region", "PMEP coastal region."),
        ("sites.csv", "estuary_area_ha", "PMEP estuary area in hectares."),
        ("sites.csv", "mouth_lat", "Study mouth/AOI latitude in decimal degrees."),
        ("sites.csv", "mouth_lon", "Study mouth/AOI longitude in decimal degrees."),
        ("sites.csv", "has_pmep_match", "Whether the study site matched a PMEP record."),
        ("sites.csv", "has_usgs_match", "Whether a USGS station was matched to the site."),
        ("aoi_polygons.geojson", "geometry", "AOI polygon used to crop imagery for each site."),
        ("labels_*.csv", "image_id", "Unique image observation ID: site_id plus asset_id."),
        ("labels_*.csv", "asset_id", "Planet or SkySat asset identifier from source imagery."),
        ("labels_*.csv", "label", "Cleaned public label: closed, open, partial, or low_quality."),
        ("labels_*.csv", "raw_label", "Original label value before public-label cleanup."),
        ("labels_*.csv", "label_set", "Label product: planetscope, skysat, or planetscope_water."),
        ("labels_*.csv", "sample_source", "Internal sample source for PlanetScope labels."),
        (
            "predictions_image_level.csv",
            "asset_id",
            "Planet asset identifier from the source image.",
        ),
        (
            "predictions_image_level.csv",
            "p_open",
            "Image-level predicted probability of open state.",
        ),
        (
            "predictions_image_level.csv",
            "p_low_quality",
            "Image-level predicted probability that the image is low quality or uncertain.",
        ),
        (
            "predictions_image_level.csv",
            "low_quality_flag",
            "Whether the source image-level prediction was marked low quality/uncertain.",
        ),
        (
            "predictions_image_level.csv",
            "predicted_state",
            "Image-level state from y_pred and y_pred_unsure in the source prediction file.",
        ),
        (
            "predictions_daily.csv",
            "p_open",
            "Daily open probability from the merged daily prediction file; blank for "
            "low-quality rows when blank in the source file.",
        ),
        (
            "predictions_daily.csv",
            "acquired_utc",
            "Acquisition time retained in the merged daily prediction file.",
        ),
        (
            "predictions_daily.csv",
            "representative_image_filename",
            "Representative image filename retained in the merged daily prediction file.",
        ),
        (
            "predictions_daily.csv",
            "asset_id",
            "Planet asset identifier for the representative image in the merged daily file.",
        ),
        (
            "predictions_daily.csv",
            "low_quality_flag",
            "Whether the merged daily prediction was marked low quality/uncertain.",
        ),
        (
            "predictions_daily.csv",
            "predicted_state",
            "Daily state from y_pred and y_pred_unsure in the merged daily prediction file.",
        ),
    ]
    pd.DataFrame(rows, columns=["file", "column", "description"]).to_csv(path, index=False)


def write_license(path: Path) -> None:
    text = f"""# License

This dataset is licensed under the {LICENSE_NAME} license ({LICENSE_SPDX}).

License URL: {LICENSE_URL}

You are free to share and adapt the dataset with appropriate attribution. This package
does not redistribute PlanetScope or SkySat imagery; `asset_id` values identify source
imagery used to create labels and predictions.
"""
    path.write_text(text)


def write_zenodo_metadata(path: Path) -> None:
    citation = (
        f"Dorman, K., & Cavanaugh, K. ({pd.Timestamp.now().year}). "
        f"{DATASET_TITLE} (Version {DATASET_VERSION}) [Data set]. Zenodo. "
        f"{DATASET_DOI_URL}"
    )
    text = f"""# Zenodo metadata

## Upload type

Dataset

## Title

{DATASET_TITLE}

## Creators

- Kyle Dorman; ORCID: 0009-0008-6969-0333; affiliation: Department of
  Geography, University of California - Los Angeles
- Kyle Cavanaugh; affiliation: Department of Geography, University of
  California - Los Angeles

## Version

{DATASET_VERSION}

## Description

This dataset contains site metadata, area-of-interest polygons, image labels,
image-level model predictions, and daily connectivity predictions for 66 intermittently
closed estuaries along the California coast from 2017-2025. Labels are provided
separately for PlanetScope imagery, SkySat validation imagery, and PlanetScope imagery
interpreted with water-level context. Predictions include image-level open-state
probabilities, image-level low-quality probabilities, and merged daily connectivity
states.

Satellite imagery is not included. PlanetScope and SkySat source images are identified
using `asset_id` values.

## Keywords

intermittently closed estuaries; estuary connectivity; PlanetScope; SkySat; California
coast; remote sensing; machine learning

## License

{LICENSE_NAME} ({LICENSE_SPDX})

## Related identifiers

- Code repository: {CODE_REPOSITORY_URL}
- Related manuscript: {MANUSCRIPT_TITLE}

## Suggested citation

{citation}

## Contact

{CONTACT_EMAIL}
"""
    path.write_text(text)


def write_readme(
    path: Path,
    counts: dict[str, int],
    missing_pmep_sites: list[str],
) -> None:
    citation = (
        f"Dorman, K., & Cavanaugh, K. ({pd.Timestamp.now().year}). "
        f"{DATASET_TITLE} (Version {DATASET_VERSION}) [Data set]. Zenodo. "
        f"{DATASET_DOI_URL}"
    )
    text = f"""# {DATASET_TITLE}

This directory contains cleaned data products for the California estuary connectivity study.

## Citation

{citation}

## Related resources

- Manuscript: {MANUSCRIPT_TITLE}
- Code repository: {CODE_REPOSITORY_URL}
- Contact: {CONTACT_EMAIL}

## License

This dataset is licensed under the {LICENSE_NAME} license ({LICENSE_SPDX}).
See `LICENSE.txt` for details.

## Files

- `sites.csv`: cleaned per-site metadata for the 66 estuaries analyzed in the manuscript.
- `sites.geojson`: point version of the cleaned site metadata.
- `aoi_polygons.geojson`: merged AOI/crop polygons used for image extraction.
- `labels_planetscope.csv`: PlanetScope labels used for model training/evaluation.
- `labels_skysat.csv`: SkySat validation labels.
- `labels_planetscope_water.csv`: PlanetScope labels assigned with water-level context.
- `predictions_image_level.csv`: image-level PlanetScope model predictions.
- `predictions_daily.csv`: cleaned merged daily PlanetScope model predictions.
- `data_dictionary.csv`: column definitions.
- `manifest.csv`: output row counts.
- `zenodo_metadata.md`: copy/paste metadata for the Zenodo upload form.
- `LICENSE.txt`: dataset license summary.

## Label cleanup

Original labels are retained in `raw_label`. Public labels use:

- `closed`
- `open`
- `partial` for original `perched open`
- `low_quality` for original `unsure`

The three label sets are intentionally kept separate because they represent different
sensor/interpretation contexts.

## Prediction cleanup

Image-level predictions retain `p_open` and `p_low_quality`. Image-level
`low_quality_flag` and `predicted_state` are derived from the source file's `y_pred`
and `y_pred_unsure` columns, preserving the decisions already made by the analysis
pipeline. The original absolute local image paths are removed and replaced with
`image_id` and `image_filename`.

Daily predictions are cleaned from the merged daily prediction file used in the analysis.
Daily `predicted_state` is derived from the source file's `y_pred` and `y_pred_unsure`
columns. This preserves the binary argmax convention used by the analysis pipeline.
Daily `p_open` is blank for rows already marked low quality in the source merged daily
file when no open probability was retained.

All label and prediction files include `asset_id`, which is the direct Planet/SkySat
source image identifier.

Satellite imagery is not included in this package.

## Counts

"""
    for name, count in counts.items():
        text += f"- `{name}`: {count:,} rows\n"

    text += """
## Notes

"""
    if missing_pmep_sites:
        text += (
            "PMEP fields are blank for "
            f"{', '.join(missing_pmep_sites)}; these sites are retained in the dataset.\n"
        )
    else:
        text += "All sites have matched PMEP fields in the source metadata.\n"

    path.write_text(text)


def write_manifest(path: Path, counts: dict[str, int]) -> None:
    pd.DataFrame([{"file": name, "rows": count} for name, count in counts.items()]).to_csv(
        path, index=False
    )


@click.command()
@click.option("--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR)
@click.option("--metadata-path", type=click.Path(path_type=Path), default=DEFAULT_METADATA_PATH)
@click.option("--aoi-dir", type=click.Path(path_type=Path), default=DEFAULT_AOI_DIR)
@click.option(
    "--skysat-labels-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_SKYSAT_LABELS_PATH,
)
@click.option(
    "--planetscope-labels-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_PLANETSCOPE_LABELS_PATH,
)
@click.option(
    "--planetscope-water-labels-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_PLANETSCOPE_WATER_LABELS_PATH,
)
@click.option(
    "--image-predictions-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_IMAGE_PREDS_PATH,
)
@click.option(
    "--daily-predictions-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_DAILY_PREDS_PATH,
)
def main(
    output_dir: Path,
    metadata_path: Path,
    aoi_dir: Path,
    skysat_labels_path: Path,
    planetscope_labels_path: Path,
    planetscope_water_labels_path: Path,
    image_predictions_path: Path,
    daily_predictions_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    site_df, sites_gdf = clean_sites(metadata_path)
    valid_site_ids = set(site_df["site_id"].astype(int))

    missing_site_names = sorted(site_df.loc[site_df["site_name"].isna(), "site_id"].tolist())
    if missing_site_names:
        click.echo(f"Warning: missing site_name for site_id values: {missing_site_names}")

    missing_pmep = sorted(site_df.loc[~site_df["has_pmep_match"], "site_id"].tolist())
    if missing_pmep:
        click.echo(f"PMEP fields are blank for site_id values: {missing_pmep}")

    sites_csv = output_dir / "sites.csv"
    site_df.to_csv(sites_csv, index=False)
    write_geojson(sites_gdf, output_dir / "sites.geojson")

    aoi_gdf = merge_aoi_polygons(aoi_dir, site_df)
    write_geojson(aoi_gdf, output_dir / "aoi_polygons.geojson")

    planetscope_labels, removed_planetscope = clean_label_file(
        planetscope_labels_path,
        valid_site_ids,
        label_set="planetscope",
    )
    skysat_labels, removed_skysat = clean_label_file(
        skysat_labels_path,
        valid_site_ids,
        label_set="skysat",
    )
    water_labels, removed_water = clean_label_file(
        planetscope_water_labels_path,
        valid_site_ids,
        label_set="planetscope_water",
    )

    if removed_planetscope:
        click.echo(f"Filtered {removed_planetscope} PlanetScope label rows outside analysis sites.")
    if removed_skysat:
        click.echo(f"Filtered {removed_skysat} SkySat label rows outside analysis sites.")
    if removed_water:
        click.echo(f"Filtered {removed_water} PlanetScope+water label rows outside analysis sites.")

    planetscope_labels.to_csv(output_dir / "labels_planetscope.csv", index=False)
    skysat_labels.to_csv(output_dir / "labels_skysat.csv", index=False)
    water_labels.to_csv(output_dir / "labels_planetscope_water.csv", index=False)

    image_predictions = clean_image_predictions(
        image_predictions_path,
        valid_site_ids,
    )
    daily_predictions = clean_daily_predictions(
        daily_predictions_path,
        valid_site_ids,
    )

    image_predictions.to_csv(output_dir / "predictions_image_level.csv", index=False)
    daily_predictions.to_csv(output_dir / "predictions_daily.csv", index=False)

    counts = {
        "sites.csv": len(site_df),
        "sites.geojson": len(sites_gdf),
        "aoi_polygons.geojson": len(aoi_gdf),
        "labels_planetscope.csv": len(planetscope_labels),
        "labels_skysat.csv": len(skysat_labels),
        "labels_planetscope_water.csv": len(water_labels),
        "predictions_image_level.csv": len(image_predictions),
        "predictions_daily.csv": len(daily_predictions),
    }
    write_data_dictionary(output_dir / "data_dictionary.csv")
    write_manifest(output_dir / "manifest.csv", counts)
    write_license(output_dir / "LICENSE.txt")
    write_zenodo_metadata(output_dir / "zenodo_metadata.md")
    missing_pmep_sites = [
        f"{int(row.site_id)} ({row.site_name})"
        for row in site_df.loc[
            ~site_df["has_pmep_match"],
            ["site_id", "site_name"],
        ].itertuples(index=False)
    ]
    write_readme(
        output_dir / "README.md",
        counts=counts,
        missing_pmep_sites=missing_pmep_sites,
    )

    click.echo(f"Wrote cleaned shared data to {output_dir}")


if __name__ == "__main__":
    main()
