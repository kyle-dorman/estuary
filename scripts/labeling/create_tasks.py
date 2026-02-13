#!/usr/bin/env python
import base64
import json
from pathlib import Path
from time import sleep
from typing import Any

import click
import numpy as np
import pandas as pd
import tqdm
from label_studio_sdk import LabelStudio
from PIL import Image

from estuary.util.img import tif_to_rgb


def task_created(tasks: list[dict], source_tif: Path, region: int) -> bool:
    return any(task_path_matches(t, source_tif, region) for t in tasks)


def task_path_matches(task: dict[str, dict[str, Any]], source_tif: Path, region: int) -> bool:
    task_meta = task.get("meta", {})
    task_path = task_meta.get("source_tif")
    task_region = task_meta.get("region")
    assert task_path is not None, task["meta"]
    assert task_region is not None
    return Path(task_path).stem == source_tif.stem and region == int(task_region)


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
<View style="display:flex; flex-direction:column; height:100vh; width:70vw; margin:0; padding:0;">
  <!-- Main content row -->
  <View style="flex:1; display:flex; flex-direction:row; min-height:0;">
    <!-- LEFT: title + image -->
    <View style="flex:1; display:flex; flex-direction:column; min-height:0; padding:8px 16px;">
      <!-- Title -->
      <View style="font-size:22px; font-weight:700; margin-bottom:6px;">
        <Text name="meta_text" value="Region: $region | Date: $date" />
      </View>

      <!-- Image -->
      <View style="flex:1; min-height:0;">
        <Image
          name="image"
          value="$image"
          rotateControl="false"
          zoomControl="true"
          style="width:100%; height:100%; object-fit:contain;"
        />
      </View>
    </View>
    <!-- RIGHT: vertical choices -->
    <View
      style="
        width:220px;
        padding:12px;
        border-left:1px solid #eee;
        position:sticky;
        top:0;
        align-self:flex-start;
      "
    >
      <Choices
        name="label"
        toName="image"
        choice="single-radio"
        required="true"
        layout="vertical"
      >
        <Choice value="closed" hotkey="a"/>
        <Choice value="open" hotkey="s"/>
        <Choice value="perched open" hotkey="d"/>
        <Choice value="unsure" hotkey="f"/>
      </Choices>
    </View>
  </View>
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
    "-t",
    "--to-label-path",
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="CSV of files to label.",
)
@click.option(
    "--single-project",
    is_flag=True,
    default=False,
    help="Upload all regions into a single Label Studio project instead of project per region",
)
@click.option(
    "--title",
    type=str,
    required=False,
    help="Name of labeling project.",
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
    to_label_path: Path,
    single_project: bool,
    title: str | None,
    ls_url: str,
) -> None:
    """Create a new Label Studio project and populate it with classification tasks."""
    labeling_base_dir.mkdir(exist_ok=True, parents=True)

    to_label = pd.read_csv(to_label_path)
    to_label["acquired"] = pd.to_datetime(to_label["acquired"], errors="coerce", utc=True)
    to_label = to_label.sort_values(by=["region", "acquired"])

    # Connect to Label Studio ---------------------------------------------- #
    client = LabelStudio(base_url=ls_url)

    existing_projects = {}
    project_created_tasks = {}
    for pdir in labeling_base_dir.iterdir():
        if pdir.is_dir():
            with open(pdir / "tasks.json") as f:
                tasks = json.load(f)
                existing_project_id = int(pdir.name)
                project_region = int(tasks[0]["meta"]["region"])
                project_created_tasks[existing_project_id] = tasks
                existing_projects[project_region] = existing_project_id

    global_project_id: int | None = None
    global_tasks: list = [t for ts in project_created_tasks.values() for t in ts]

    if single_project and len(project_created_tasks):
        global_project_id = int(list(project_created_tasks.keys())[0])  # type: ignore

    # Gather samples and convert to JPEGs ----------------------------------- #
    for region, rdf in tqdm.tqdm(to_label.groupby("region"), total=len(to_label.region.unique())):
        is_skysat = "skysat" in rdf["instrument"].unique()

        if single_project:
            if global_project_id is None:
                project_title = title if title is not None else "All Regions"
                project = client.projects.create(
                    title=project_title, label_config=build_label_config()
                )
                global_project_id = project.id
            project_id = global_project_id
        else:
            if region in existing_projects:
                project_id = existing_projects[region]
            else:
                if is_skysat:
                    name = "SS"
                else:
                    name = "Dove All"
                project_title = f"{name} - {region}"
                project = client.projects.create(
                    title=project_title, label_config=build_label_config()
                )
                project_id = project.id

        if single_project:
            tasks = global_tasks
        else:
            tasks = project_created_tasks.get(project_id, [])

        out_dir = labeling_base_dir / f"{project_id:05d}"
        out_dir.mkdir(exist_ok=True, parents=True)

        images_dir = out_dir / "images" / str(region)
        images_dir.mkdir(exist_ok=True, parents=True)

        created_tasks_count = 0
        for _, row in rdf.iterrows():
            # No month for skysat
            if is_skysat:
                base = base_dir / row.instrument / "results" / str(row.year) / str(region) / "files"
                tif_path: Path | None = next(
                    (p for p in base.glob(f"{row.asset_id}*_pansharpened_clip.tif")), None
                )
            else:
                base = (
                    base_dir
                    / row.instrument
                    / "results"
                    / str(row.year)
                    / str(row.month)
                    / str(region)
                    / "files"
                )
                tif_path: Path | None = next(
                    (p for p in base.glob(f"{row.asset_id}*SR*clip.tif")), None
                )

            if tif_path is None or not Path(tif_path).exists():
                continue

            if task_created(tasks, tif_path, row["region"]):
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
                "region": row["region"],
                "date": row["acquired"].strftime("%Y-%m-%d"),
                "meta": {
                    "region": region,
                    "source_tif": str(tif_path),
                    "source_jpeg": str(jpeg_path),
                },
            }

            client.tasks.create(data=task, project=project_id)
            tasks.append(task)
            created_tasks_count += 1

        if created_tasks_count:
            print("Create", created_tasks_count, "for region", region)

        (out_dir / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        sleep(5)


if __name__ == "__main__":
    main()
