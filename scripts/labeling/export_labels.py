#!/usr/bin/env python
"""Export labels from a group of Label Studio projects to a plain text file:

<image_path> <label>

If multiple annotations or choices exist, the first choice of the first
annotation is used. Assumes tasks were created with `meta.source_tif`.
"""

import os
from pathlib import Path

import click
import pandas as pd
from label_studio_sdk.client import LabelStudio
from tqdm import tqdm

from estuary.model.data import parse_dt_from_pth


def extract_label(task: dict) -> str | None:
    """Return the first classification choice from the task, or None."""
    annotations = task.get("annotations") or []
    for ann in annotations:
        for r in ann.get("result", []):
            if r.get("value", {}).get("choices"):
                return r["value"]["choices"][0]
    return None


@click.command()
@click.option(
    "-d",
    "--labeling-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory with label studio projects.",
)
@click.option(
    "-rd",
    "--regions-dir",
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Directory under which region geojsons are stored.",
)
@click.option(
    "--out", required=True, type=click.Path(writable=True, path_type=Path), help="Output .csv file."
)
@click.option(
    "--ls-url",
    default="http://localhost:8080",
    show_default=True,
    help="Label Studio base URL.",
)
def main(labeling_dir: Path, out: Path, regions_dir: Path, ls_url: str):
    # Must set env key LABEL_STUDIO_API_KEY
    ls = LabelStudio(base_url=ls_url)

    rows = []
    dirs = list(labeling_dir.iterdir())
    valid_regions = set(int(p.stem) for p in regions_dir.glob("*.geojson"))
    for pdir in tqdm(dirs):
        if not pdir.is_dir():
            continue
        project_id = int(pdir.stem)
        project = ls.projects.get(project_id)
        assert project.id is not None
        tasks = ls.projects.exports.as_json(project.id)

        if not tasks:
            click.echo("No labeled tasks found.")
            return

        for task in tasks:
            label = extract_label(task)
            if not label:
                continue
            # meta may be stored at top-level or nested under data
            meta = task.get("data", {}).get("meta") or {}
            region = meta.get("region")
            if int(region) not in valid_regions:
                continue
            source_tif = Path(meta["source_tif"])
            if not source_tif.exists():
                print(f"{source_tif} DOESNT EXIST. Skipping...")
                continue
            acquired = parse_dt_from_pth(source_tif)
            instrument = "skysat" if "skysat" in str(source_tif) else source_tif.parents[5].name
            rows.append(
                {
                    "region": region,
                    "source_tif": source_tif,
                    "label": label,
                    "acquired": acquired,
                    "instrument": instrument,
                }
            )

    if not rows:
        click.echo("No labels extracted.")
        return

    df = pd.DataFrame(rows)
    df["acquired"] = pd.to_datetime(df["acquired"], errors="coerce")
    if out.exists():
        os.remove(out)
    df.to_csv(out, index=False)
    click.echo(f"Exported {len(df)} labels to {out}")


if __name__ == "__main__":
    main()
