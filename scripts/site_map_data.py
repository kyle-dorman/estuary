"""Produce the public map subset from the pinned Zenodo archive, using stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site-map"
ARCHIVE_NAME = "california_estuary_connectivity_shared_data_v1.0.0.zip"
ARCHIVE_MD5 = "c6ca683de52019987dd1c13c967a5cb2"
ARCHIVE_SHA256 = "2f9a2386a64b9c1d3834a37fa8d50e24ac71429f0dcd16e602762fdf2d626596"
SOURCE_MEMBER = "shared_data/sites.geojson"
SOURCE_SHA256 = "4107ce5b025a0e0147a06ac268e39b510aba81a8beb8ad96bfe750be466dbc8b"
OUTPUT_SHA256 = "0436734088e92a75b895cb747696b35d0d4587ebcf8f814763bc0616f5333286"
FIELDS = ("site_id", "site_name", "pmep_region", "cmecs_class", "estuary_area_ha")
EXPECTED_IDS = (
    11,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    25,
    27,
    28,
    29,
    31,
    32,
    33,
    34,
    35,
    38,
    39,
    43,
    44,
    46,
    48,
    50,
    51,
    54,
    56,
    57,
    59,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    70,
    72,
    77,
    79,
    81,
    83,
    84,
    85,
    86,
    88,
    92,
    93,
    94,
    95,
    96,
    98,
    2138,
    2145,
    2147,
    12097,
    12103,
    13008,
    13009,
    13027,
    13057,
    13073,
    13099,
    14054,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate(data: dict) -> None:
    """Reject malformed, private, missing, duplicate, or implausibly located sites."""
    require(isinstance(data, dict), "GeoJSON must be an object")
    require(set(data) == {"type", "features"}, "Unexpected collection fields")
    require(data["type"] == "FeatureCollection", "Expected a FeatureCollection")
    features = data["features"]
    require(isinstance(features, list) and len(features) == 66, "Expected exactly 66 sites")
    ids, names = [], []
    for feature in features:
        require(isinstance(feature, dict), "Feature must be an object")
        require(
            set(feature) == {"type", "id", "properties", "geometry"}, "Unexpected feature fields"
        )
        require(feature["type"] == "Feature", "Expected a Feature")
        props = feature["properties"]
        require(
            isinstance(props, dict) and set(props) == set(FIELDS), "Property allowlist mismatch"
        )
        site_id = props["site_id"]
        require(type(site_id) is int and site_id > 0, "site_id must be a positive integer")
        require(type(feature["id"]) is int and feature["id"] == site_id, "Feature ID mismatch")
        name = props["site_name"]
        require(
            isinstance(name, str) and bool(name.strip()) and name == name.strip(), "Invalid name"
        )
        for field in ("pmep_region", "cmecs_class"):
            value = props[field]
            require(
                value is None or (isinstance(value, str) and bool(value.strip())),
                f"Invalid {field}",
            )
        area = props["estuary_area_ha"]
        require(area is None or (number(area) and area >= 0), "Invalid area")
        geometry = feature["geometry"]
        require(
            isinstance(geometry, dict) and set(geometry) == {"type", "coordinates"},
            "Invalid geometry",
        )
        require(geometry["type"] == "Point", "Only Point geometry is supported")
        coords = geometry["coordinates"]
        require(
            isinstance(coords, list) and len(coords) == 2 and all(number(c) for c in coords),
            "Expected finite [longitude, latitude]",
        )
        require(
            -125 <= coords[0] <= -114 and 32 <= coords[1] <= 43,
            "Coordinates outside California bounds or axes reversed",
        )
        ids.append(site_id)
        names.append(name.casefold())
    require(len(set(ids)) == 66, "Duplicate site IDs")
    require(len(set(names)) == 66, "Duplicate site names")
    require(
        tuple(ids) == EXPECTED_IDS, "Site inventory or stable ordering differs from version 1.0.0"
    )


def provenance(output_bytes: bytes) -> dict:
    return {
        "dataset_title": "Data for: Seasonal and interannual dynamics of estuary connectivity "
        "along the California coast",
        "dataset_version": "1.0.0",
        "dataset_doi": "10.5281/zenodo.20753031",
        "dataset_url": "https://doi.org/10.5281/zenodo.20753031",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Dorman, K., Largier, J. L., & Cavanaugh, K. C. (2026). "
        "Data for: Seasonal and interannual "
        "dynamics of estuary connectivity along the California coast "
        "(Version 1.0.0) [Data set]. Zenodo.",
        "archive_filename": ARCHIVE_NAME,
        "archive_url": f"https://zenodo.org/records/20753031/files/{ARCHIVE_NAME}?download=1",
        "archive_md5": ARCHIVE_MD5,
        "archive_sha256": ARCHIVE_SHA256,
        "source_member": SOURCE_MEMBER,
        "source_sha256": SOURCE_SHA256,
        "source_preparation_script": "scripts/share_data/prepare_shared_data.py:clean_sites",
        "generation_script": "scripts/site_map_data.py",
        "selection": "All 66 archived sites; only the listed public properties are retained. "
        "Point coordinates and numeric site_id values are unchanged. "
        "GeoJSON feature.id is copied from site_id.",
        "properties": list(FIELDS),
        "coordinate_convention": "WGS 84 (EPSG:4326); GeoJSON [longitude, latitude] in decimal "
        "degrees (CRS84 axis order). Points represent study mouth/AOI "
        "locations, not estuary boundaries or current mouth positions.",
        "site_count": 66,
        "site_ids": list(EXPECTED_IDS),
        "output_filename": "estuary-sites.geojson",
        "output_sha256": sha256(output_bytes),
    }


def generate(source_zip: Path, output_dir: Path = OUTPUT) -> None:
    archive = source_zip.read_bytes()
    require(sha256(archive) == ARCHIVE_SHA256, "Archive SHA256 differs from pinned Zenodo release")
    with zipfile.ZipFile(source_zip) as zipped:
        source = zipped.read(SOURCE_MEMBER)
    require(sha256(source) == SOURCE_SHA256, "Archived sites SHA256 mismatch")
    raw = json.loads(source)
    features = []
    for feature in raw["features"]:
        props = feature["properties"]
        require(
            feature["geometry"]["coordinates"] == [props["mouth_lon"], props["mouth_lat"]],
            "Archived mouth coordinate mismatch",
        )
        features.append(
            {
                "type": "Feature",
                "id": props["site_id"],
                "properties": {field: props[field] for field in FIELDS},
                "geometry": feature["geometry"],
            }
        )
    features.sort(key=lambda feature: feature["id"])
    result = {"type": "FeatureCollection", "features": features}
    validate(result)
    output_bytes = encode(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "estuary-sites.geojson").write_bytes(output_bytes)
    (output_dir / "provenance.json").write_bytes(encode(provenance(output_bytes)))
    check(output_dir)


def check(output_dir: Path = OUTPUT) -> None:
    output_bytes = (output_dir / "estuary-sites.geojson").read_bytes()
    data = json.loads(output_bytes)
    validate(data)
    require(output_bytes == encode(data), "GeoJSON does not use deterministic serialization")
    require(sha256(output_bytes) == OUTPUT_SHA256, "GeoJSON differs from pinned release subset")
    expected = encode(provenance(output_bytes))
    require((output_dir / "provenance.json").read_bytes() == expected, "Provenance mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true", help="Validate committed data offline")
    actions.add_argument(
        "--source-zip", type=Path, help="Regenerate from the pinned Zenodo archive"
    )
    args = parser.parse_args()
    try:
        if args.source_zip:
            generate(args.source_zip)
        else:
            check()
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Site data validation failed: {error}\n")
    print("Validated 66 unique estuaries, coordinates, public properties, and provenance.")


if __name__ == "__main__":
    main()
