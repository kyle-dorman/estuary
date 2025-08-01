#!/usr/bin/env python
import base64
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import click
import tqdm
from label_studio_sdk import LabelStudio
from PIL import Image

from estuary.util import tif_to_rgb

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
        --n 30 \
        --ls-url http://localhost:8080 \
        --ls-token <YOUR_API_TOKEN>

Environment
-----------
* `LABEL_STUDIO_API_TOKEN` can be used instead of --ls-token.
"""

SKIP_REGION = ["arroyo_sequit", "drakes_estero", "eel_river", "mugu_lagoon", "batiquitos_lagoon"]


# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #


def iter_regions(base_dir: Path) -> list[tuple[str, list[Path]]]:
    """Walk BASE/YEAR/MONTH/REGION/files/*_SR_clip.tif and return
    [(region_name, [tif, tif, ...]), ...]
    """
    regions = {}
    for tif in base_dir.glob("*/*/*/files/*_SR_clip.tif"):
        # e.g. base/2024/06/my_region/files/scene_SR_clip.tif --> "my_region"
        region = tif.parents[1].name
        if region in SKIP_REGION:
            continue
        regions.setdefault(region, []).append(tif)
    return list(regions.items())


def sample_paths(region_paths: list[Path], n: int, rng: random.Random) -> list[Path]:
    """Pick up to *n* paths at random."""
    if len(region_paths) <= n:
        return region_paths
    return rng.sample(region_paths, n)


def tif_to_jpeg(
    tif_path: Path,
    dest_path: Path,
    crop_size: tuple[int, int, int, int] | None,
    jpeg_quality: int = 95,
) -> None:
    """Convert a GeoTIFF to RGB JPEG using estuary.util.tif_to_rgb()."""
    rgb = tif_to_rgb(tif_path)
    img = Image.fromarray(rgb).resize((512, 512))
    if crop_size is not None:
        img = img.crop(crop_size)
    img.save(dest_path, format="JPEG", quality=jpeg_quality, optimize=True)


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
      <Choice value="unsure" hotkey="d"/>
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
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    required=True,
    help="Root directory with YEAR/MONTH/REGION/files/*.tif structure.",
)
@click.option(
    "-ld",
    "--labeling-base-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    required=True,
    help="Directory under which JPEGs + tasks.json will be stored.",
)
@click.option(
    "-n",
    "--num-per-region",
    type=int,
    default=10,
    show_default=True,
    help="Number of random images to sample per region.",
)
@click.option(
    "--ls-url",
    default="http://localhost:8080",
    show_default=True,
    help="Base URL of local Label Studio instance.",
)
@click.option(
    "--ls-token",
    envvar="LABEL_STUDIO_API_TOKEN",
    required=True,
    help="API token for Label Studio (env LABEL_STUDIO_API_TOKEN also works).",
)
@click.option("--seed", default=42, show_default=True, help="RNG seed.")
def main(
    base_dir: str,
    labeling_base_dir: str,
    num_per_region: int,
    ls_url: str,
    ls_token: str,
    seed: int,
) -> None:
    """Create a new Label Studio project and populate it with classification tasks."""
    base_path = Path(base_dir)
    labeling_base_path = Path(labeling_base_dir)
    rng = random.Random(seed)

    click.echo(f"Scanning regions under {base_path} …")
    region_items = iter_regions(base_path)
    if not region_items:
        click.echo("No regions found!", err=True)
        sys.exit(1)

    with open(labeling_base_path / "region_crops.json") as f:
        region_crops = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Connect to Label Studio ---------------------------------------------- #
    client = LabelStudio(base_url=ls_url, api_key=ls_token)
    project_title = f"Estuary-{timestamp}"
    project = client.projects.create(title=project_title, label_config=build_label_config())
    project_id = project.id
    out_dir = labeling_base_path / f"{project_id:05d}"
    out_dir.mkdir(parents=True, exist_ok=False)

    # set up local export storage so annotations land in the same folder
    client.import_storage.local.create(
        title="LocalExport", path=str(out_dir), use_blob_urls=False, project=project_id
    )

    # Gather samples and convert to JPEGs ----------------------------------- #
    click.echo(f"Created LS project {project_id} – importing tasks …")
    tasks = []
    for region, paths in tqdm.tqdm(region_items):
        chosen = sample_paths(paths, num_per_region, rng)
        if not chosen:
            continue

        region_dir = out_dir / region / "images"
        region_dir.mkdir(exist_ok=True, parents=True)
        crop = region_crops.get(region, None)

        for tif_path in chosen:
            jpeg_name = tif_path.stem.replace("_SR_clip", "") + ".jpg"
            jpeg_path = region_dir / jpeg_name
            tif_to_jpeg(tif_path, jpeg_path, crop)

            task = {
                "image": image_to_datauri(jpeg_path),
                "meta": {
                    "region": region,
                    "source_tif": str(tif_path),
                    "source_jpeg": str(jpeg_path),
                },
            }

            client.tasks.create(
                project=project_id,
                data=task,
            )

            tasks.append(task)

    if not tasks:
        click.echo("Nothing sampled; aborting.", err=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        sys.exit(1)

    click.echo(f"Prepared {len(tasks)} tasks in {out_dir}")

    # Save tasks.json for reproducibility
    (out_dir / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    click.echo(
        f"✅ All set!  LS URL: {ls_url.rstrip('/')}/projects/{project_id} "
        f"(images & labels in {out_dir})"
    )


if __name__ == "__main__":
    main()
