#!/usr/bin/env python
"""Export labels from a Label Studio project to a plain text file:

<image_path> <label>

If multiple annotations or choices exist, the first choice of the first
annotation is used. Assumes tasks were created with `meta.source_tif`.
"""

from pathlib import Path

import click
import pandas as pd
from label_studio_sdk.client import LabelStudio


def extract_label(task: dict) -> str | None:
    """Return the first classification choice from the task, or None."""
    annotations = task.get("annotations") or []
    for ann in annotations:
        for r in ann.get("result", []):
            if r.get("value", {}).get("choices"):
                return r["value"]["choices"][0]
    return None


@click.command()
@click.option("-p", "--project-id", required=True, type=int, help="Label Studio project ID.")
@click.option("--out", required=True, type=click.Path(writable=True), help="Output .txt file.")
@click.option(
    "--ls-url",
    default="http://localhost:8080",
    show_default=True,
    help="Label Studio base URL.",
)
@click.option(
    "--ls-token",
    envvar="LABEL_STUDIO_API_TOKEN",
    required=True,
    help="Label Studio API token (or set LABEL_STUDIO_API_TOKEN).",
)
def main(project_id: int, out: str, ls_url: str, ls_token: str):
    ls = LabelStudio(base_url=ls_url, api_key=ls_token)
    project = ls.projects.get(project_id)
    assert project.id is not None
    tasks = ls.projects.exports.as_json(project.id)

    if not tasks:
        click.echo("No labeled tasks found.")
        return

    rows = []
    for task in tasks:
        label = extract_label(task)
        if not label:
            continue
        # meta may be stored at top-level or nested under data
        meta = task.get("data", {}).get("meta") or {}
        region = meta.get("region")
        source_tif = meta.get("source_tif")
        source_jpeg = meta.get("source_jpeg")
        if not source_tif:
            continue
        rows.append(
            {
                "region": region,
                "source_tif": source_tif,
                "source_jpeg": source_jpeg,
                "label": label,
            }
        )

    if not rows:
        click.echo("No labels extracted.")
        return

    df = pd.DataFrame(rows)
    out_path = Path(out)
    df.to_csv(out_path, index=False)
    click.echo(f"Exported {len(df)} labels to {out_path}")


if __name__ == "__main__":
    main()
