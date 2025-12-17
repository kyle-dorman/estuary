#!/usr/bin/env python
import base64
import json
from pathlib import Path

import click
import pandas as pd
import tqdm
from label_studio_sdk import LabelStudio


def image_to_datauri(jpeg_path: Path) -> str:
    """Return a data URI string for an image (JPEG)."""
    b64 = base64.b64encode(jpeg_path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_label_config() -> str:
    """XML view with three classification labels."""
    return """
<View style="display:flex; flex-direction:column; width:100%; box-sizing:border-box;">
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

    # set up local export storage so annotations land in the same folder
    client.import_storage.local.create(
        title="LocalExport", path=str(out_dir), use_blob_urls=False, project=project_id
    )

    for _, row in tqdm.tqdm(to_label.iterrows(), total=len(to_label)):
        created_tasks_count = 0

        source_tif = row["source_tif"]

        if any(d["meta"]["source_tif"] == source_tif for d in created_tasks):
            continue

        task = {
            "image": image_to_datauri(Path(row["img_path"])),
            "plot_image": image_to_datauri(Path(row["fig_path"])),
            "meta": {
                "region": row["region"],
                "source_tif": source_tif,
                "orig_label": row["orig_label"],
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
