#!/usr/bin/env python
import base64
import datetime
import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
import tqdm
from label_studio_sdk import LabelStudio


def task_created(tasks: list[dict], source_tif: Path, region: int) -> bool:
    return any(task_path_matches(t, source_tif, region) for t in tasks)


def task_path_matches(task: dict[str, dict[str, Any]], source_tif: Path, region: int) -> bool:
    task_meta = task.get("meta", {})
    task_path = task_meta.get("source_tif")
    task_region = task_meta.get("region")
    assert task_path is not None, task["meta"]
    assert task_region is not None
    return Path(task_path).stem == source_tif.stem and region == int(task_region)


def image_to_datauri(jpeg_path: Path) -> str:
    """Return a data URI string for an image (JPEG)."""
    b64 = base64.b64encode(jpeg_path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_label_config() -> str:
    """XML view with three classification labels."""
    return """
<View style="display:flex; flex-direction:column; width:100%; box-sizing:border-box;">
  <!-- Metadata header -->
  <View style="padding: 0.25em 0.5em; font-size: 14px;">
    <Text name="meta_text" value="Region: $region | Date: $date" />
  </View>

  <View style="position: sticky; top: 0; z-index: 10; padding: 0.5em 0;">
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

  <!-- Images row -->
  <View
    style="display:flex; flex-direction:row; width:100%; gap:12px; box-sizing:border-box;
    align-items:flex-start;">
    <View style="flex:1; min-width:0;">
      <Image
        name="image" value="$image" rotateControl="false" style="width:100%;"
        brightnessControl="true" contrastControl="true"/>
    </View>

    <View style="flex:1; min-width:0;">
      <Image
        name="plot_image" value="$plot_image" rotateControl="false" zoomControl="false"
        style="width:100%;"/>
    </View>
  </View>
</View>
""".strip()


# --------------------------------------------------------------------------- #
#                                 CLI entry                                   #
# --------------------------------------------------------------------------- #


@click.command()
@click.option(
    "-l",
    "--to-label-path",
    type=click.Path(exists=True, file_okay=True, resolve_path=True, path_type=Path),
    required=True,
    help="Path to the csv of tasks to create.",
)
@click.option(
    "-d",
    "--labeling-base-dir",
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Directory under which JPEGs + tasks.json will be stored.",
)
@click.option(
    "--title",
    type=str,
    required=True,
    help="Name of labeling project.",
)
@click.option(
    "--ls-url",
    default="http://localhost:8080",
    show_default=True,
    help="Base URL of local Label Studio instance.",
)
def main(
    to_label_path: Path,
    labeling_base_dir: Path,
    title: str,
    ls_url: str,
) -> None:
    """Create a new Label Studio project and populate it with classification tasks."""
    labeling_base_dir.mkdir(exist_ok=True, parents=True)

    to_label = pd.read_csv(to_label_path)
    to_label["acquired"] = to_label.source_tif.apply(
        lambda f: datetime.datetime.strptime(Path(f).stem.split("_")[0], "%Y%m%d")
    )
    to_label = to_label.sort_values(by=["region", "acquired"])

    # Connect to Label Studio ---------------------------------------------- #
    client = LabelStudio(base_url=ls_url)

    project_id: int | None = None
    created_tasks = []
    for pdir in labeling_base_dir.iterdir():
        if pdir.is_dir():
            with open(pdir / "tasks.json") as f:
                created_tasks.extend(json.load(f))
                assert project_id is None, "Just one project!"
                project_id = int(pdir.name)

    if project_id is None:
        project = client.projects.create(title=title, label_config=build_label_config())
        project_id = project.id

    out_dir = labeling_base_dir / f"{project_id:05d}"
    out_dir.mkdir(exist_ok=True, parents=True)

    created_tasks_count = 0
    for _, row in tqdm.tqdm(to_label.iterrows(), total=len(to_label)):
        source_tif = row["source_tif"]
        if any(
            not p.exists() for p in [Path(source_tif), Path(row["img_path"]), Path(row["fig_path"])]
        ):
            continue
        if task_created(created_tasks, Path(source_tif), row["region"]):
            continue

        task = {
            "image": image_to_datauri(Path(row["img_path"])),
            "plot_image": image_to_datauri(Path(row["fig_path"])),
            "region": row["region"],
            "date": row["acquired"].strftime("%Y-%m-%d"),
            "meta": {
                "region": row["region"],
                "source_tif": source_tif,
                "source_jpeg": row["img_path"],
                "fig_jpeg": row["fig_path"],
            },
        }

        client.tasks.create(data=task, project=project_id)
        created_tasks.append(task)
        created_tasks_count += 1

    if created_tasks_count:
        print("Created", created_tasks_count, "tasks")

    # Save tasks.json for reproducibility
    (out_dir / "tasks.json").write_text(json.dumps(created_tasks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
