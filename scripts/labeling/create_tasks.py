#!/usr/bin/env python
import base64
import gc
import json
import sys
import time
from pathlib import Path

import click
import numpy as np
import tqdm
from label_studio_sdk import LabelStudio
from PIL import Image

from estuary.model.data import parse_dt_from_pth
from estuary.util.img import tif_to_rgb

"""
create_tasks.py

Discover GeoTIFF scenes in the estuary data hierarchy, sample *N* random
*_SR_clip.tif files **per region**, convert them to RGB JPEGs, and upload the
resulting tasks to a local Label Studio instance, creating a brand-new project
on each run.

Directory assumptions
---------------------
SOURCE: BASE/YEAR/MONTH/REGION/files/*_SR_clip.tif
TARGET: LABELING_BASE_DIR/<project_id>/   (auto-created)

Each generated task asks the annotator to choose **open / closed / unsure**.

Usage
-----
    python scripts/create_labeling_task.py \
        --base-dir /data/estuary \
        --labeling-base-dir /data/label_runs \
        --ls-url http://localhost:8080 \
        --ls-token <YOUR_API_TOKEN>

Environment
-----------
* `LABEL_STUDIO_API_KEY` must be set.
"""


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #


def regions_tifs(base_dir: Path, valid_regions: set[int]) -> dict[int, list[Path]]:
    """Walk BASE/SAT/YEAR/MONTH/REGION/files/*_SR_clip.tif and return
    [(region_name, [tif, tif, ...]), ...]
    """
    if base_dir.parent.name == "skysat":
        glob_str = "*/*/files/*_pansharpened_clip.tif"
    else:
        glob_str = "*/*/*/*/*/files/*_SR*clip.tif"

    regions = {}
    for tif in base_dir.glob(glob_str):
        # e.g. base/2024/06/my_region/files/scene_SR_clip.tif --> "my_region"
        region = int(tif.parents[1].name)
        if region not in valid_regions:
            continue
        regions.setdefault(region, []).append(tif)

    for k in regions.keys():
        regions[k] = sorted(regions[k], key=lambda p: parse_dt_from_pth(p))

    keys = sorted(list(regions.keys()))
    return {k: regions[k] for k in keys}


def tif_to_jpeg(
    tif_path: Path,
    dest_path: Path,
    jpeg_quality: int = 95,
) -> None:
    """Convert a GeoTIFF to RGB JPEG using estuary.util.tif_to_rgb()."""
    rgb = tif_to_rgb(tif_path)
    if np.all(rgb == 0):
        return
    img = Image.fromarray(rgb)
    img.save(dest_path, format="JPEG", quality=jpeg_quality, optimize=True)
    img.close()


def image_to_datauri(jpeg_path: Path) -> str:
    """Return a data URI string for an image (JPEG)."""
    b64 = base64.b64encode(jpeg_path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_label_config() -> str:
    """XML view with three classification labels."""
    return """
<View style="display: flex;">
  <View style="padding: 0em 1em; margin-right: 1em; border-radius: 3px">
    <View style="position: sticky; top: 0">
    <Choices
        name="label"
        toName="image"
        choice="single-radio"
        showInline="true"
        required="true"
        layout="inline"
    >
      <Choice value="closed" hotkey="a"/>
      <Choice value="open" hotkey="s"/>
      <Choice value="perched open" hotkey="d"/>
      <Choice value="unsure" hotkey="f"/>
    </Choices>
    </View>
  </View>
  <Image name="image" value="$image" rotateControl="false"/>
</View>
""".strip()


# --------------------------------------------------------------------------- #
#                                 CLI entry                                   #
# --------------------------------------------------------------------------- #


@click.command()
@click.option(
    "-d",
    "--base-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Root directory with SAT/YEAR/MONTH/REGION/files/*.tif structure.",
)
@click.option(
    "-ld",
    "--labeling-base-dir",
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Directory under which JPEGs + tasks.json will be stored.",
)
@click.option(
    "-rd",
    "--regions-dir",
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Directory under which region geojsons are stored.",
)
@click.option(
    "--region",
    type=int,
    required=False,
    help="Filter to a single region",
)
@click.option(
    "--ls-url",
    default="http://localhost:8080",
    show_default=True,
    help="Base URL of local Label Studio instance.",
)
def main(
    base_dir: Path,
    labeling_base_dir: Path,
    regions_dir: Path,
    region: int | None,
    ls_url: str,
) -> None:
    """Create a new Label Studio project and populate it with classification tasks."""
    labeling_base_dir.mkdir(exist_ok=True, parents=True)

    click.echo(f"Scanning regions under {base_dir} …")
    valid_regions = set(int(p.stem) for p in regions_dir.glob("*.geojson"))
    region_items = regions_tifs(base_dir, valid_regions)
    if region is not None:
        region_items = {k: v for k, v in region_items.items() if k == region}
    if not region_items:
        click.echo("No regions found!", err=True)
        sys.exit(1)

    # Connect to Label Studio ---------------------------------------------- #
    client = LabelStudio(base_url=ls_url)

    existing_projects = {}
    region_created_tasks = {}
    for pdir in labeling_base_dir.iterdir():
        if pdir.is_dir():
            with open(pdir / "tasks.json") as f:
                tasks = json.load(f)
                project_region = int(tasks[0]["meta"]["region"])
                region_created_tasks[project_region] = tasks
                existing_projects[project_region] = int(pdir.name)

    # Gather samples and convert to JPEGs ----------------------------------- #
    for run_region in tqdm.tqdm(list(region_items.keys())):
        gc.collect()
        time.sleep(0.2)
        tasks = region_created_tasks.get(run_region, [])
        created_tasks_count = 0
        paths = region_items[run_region]

        if run_region in existing_projects:
            project_id = existing_projects[run_region]
        else:
            if "skysat" in str(paths[0]):
                name = "SS"
            else:
                name = "Dove All"
            project_title = f"{name} - {run_region}"
            project = client.projects.create(title=project_title, label_config=build_label_config())
            project_id = project.id

        out_dir = labeling_base_dir / f"{project_id:05d}"
        out_dir.mkdir(exist_ok=True, parents=True)

        # set up local export storage so annotations land in the same folder
        client.import_storage.local.create(
            title="LocalExport", path=str(out_dir), use_blob_urls=False, project=project_id
        )

        images_dir = out_dir / "images"
        images_dir.mkdir(exist_ok=True, parents=True)
        for tif_path in paths:
            matches = [t for t in tasks if Path(t["meta"]["source_tif"]) == tif_path]
            if len(matches):
                continue

            if tif_path.name.endswith("pansharpened_clip.tif"):
                jpeg_name = tif_path.stem.replace("_pansharpened_clip", "") + ".jpg"
            else:
                jpeg_name = tif_path.stem.replace("_SR_clip", "") + ".jpg"
            jpeg_path = images_dir / jpeg_name
            tif_to_jpeg(tif_path, jpeg_path)

            if not jpeg_path.exists():
                continue

            task = {
                "image": image_to_datauri(jpeg_path),
                "meta": {
                    "region": run_region,
                    "source_tif": str(tif_path),
                    "source_jpeg": str(jpeg_path),
                },
            }

            client.tasks.create(data=task, project=project_id)
            tasks.append(task)
            created_tasks_count += 1
        if created_tasks_count:
            print("Create", created_tasks_count, "for region", run_region)

        # # Create all tasks
        # bulk_url = f"api/projects/{project_id}/tasks/bulk/"
        # resp = bulk_client.make_request(
        #     method="POST",
        #     url=bulk_url,
        #     json={"tasks": tasks},  # list of dicts
        # )
        # resp.raise_for_status()

        # Save tasks.json for reproducibility
        (out_dir / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
