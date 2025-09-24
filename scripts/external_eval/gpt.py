#!/usr/bin/env python
""" """

import base64
import json
import os
import sys
from collections import defaultdict
from enum import Enum
from io import BytesIO
from pathlib import Path

import click
import openai
import pandas as pd
import tqdm
from PIL import Image
from pydantic import BaseModel


class MouthLabel(str, Enum):
    UNSURE = "unsure"
    CLOSED = "closed"
    OPEN = "open"


class EstuaryResponse(BaseModel):
    label: MouthLabel
    reasoning: str


def load_prompt_cfg(path: Path) -> dict:
    """Load prompt configuration JSON (label definitions, bridge rule, etc.)."""
    with path.open() as f:
        return json.load(f)


def build_developer_text(cfg: dict) -> str:
    """Compose the developer message text from the prompt.json config."""
    parts: list[str] = []
    if cfg.get("composite_hint"):
        parts.append(cfg["composite_hint"].strip())
    # Classes block
    parts.append("\n# Classes")
    classes = cfg.get("classes", {})
    for key, meta in classes.items():
        name = key.capitalize()
        parts.append(f"\n- {name}: {meta.get('definition', '').strip()}")
        for cue in meta.get("image_cues", []):
            parts.append(f"\n  - {cue.strip()}")
    # Bridge rule
    if cfg.get("bridge_rule"):
        parts.append("\n\n## Bridge / culvert rule:")
        br = cfg["bridge_rule"]
        for text in br.values():
            parts.append(f"\n- {text.strip()}")
    # Decision flow
    if cfg.get("decision_flow"):
        parts.append("\n\n## Quick decision flow")
        for step in cfg["decision_flow"]:
            parts.append("\n" + step.strip())
    fin = "\n".join([a.strip() for a in cfg["final_instructions"]])
    parts.append(f"\n\n{fin}")
    return "".join(parts)


def b64_data_uri(img_path: Path, dim: int = 512) -> str:
    """
    Return data URI string for an image file, resized so image size is dim.
    """
    # Decide MIME based on extension, but switch to PNG if alpha is present
    mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"

    # Open and convert as needed
    img = Image.open(img_path).resize((dim, dim), resample=Image.Resampling.LANCZOS)

    assert img.mode not in ("RGBA", "LA") and not (img.mode == "P" and "transparency" in img.info)

    # Ensure suitable mode for chosen format
    if mime == "image/jpeg" and img.mode not in ("RGB",):
        img = img.convert("RGB")

    buf = BytesIO()
    if mime == "image/jpeg":
        img.save(buf, format="JPEG", quality=95, optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)

    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{data}"


def build_fewshot_messages(
    prompts: dict[str, list[str]], detail: str = "auto", dim: int = 512
) -> list[dict]:
    """build messages list."""
    prompts["unsure"] = [
        "/Users/kyledorman/data/estuary/label_studio/00025/pismo_creek_lagoon/images/20190722_185331_73_1062_3B_AnalyticMS.jpg"
    ]
    messages: list[dict] = []
    for label, example_images in prompts.items():
        if not len(example_images):
            continue
        for img_path in example_images:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Classify this image",
                        },
                        {
                            "type": "input_image",
                            "image_url": b64_data_uri(Path(img_path), dim=dim),
                            "detail": detail,
                        },
                    ],
                }
            )
            resp = EstuaryResponse(label=MouthLabel(label), reasoning="example explanation")
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(dict(resp)),
                }
            )
    return messages


def classify_image(
    client: openai.Client,
    model: str,
    fewshot: list[dict],
    img_path: Path,
    dry_run: bool,
    cfg: dict,
    detail: str = "auto",
    dim: int = 512,
) -> EstuaryResponse:
    """Send one image to the model and parse JSON reply → (label, reasoning)."""

    system_msg = {
        "role": "system",
        "content": cfg.get("system_msg"),
    }
    dev_text = build_developer_text(cfg)
    developer_msg = {
        "role": "developer",
        "content": dev_text,
    }

    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": cfg.get(
                    "user_instruction",
                    ("Classify this image"),
                ),
            },
            {
                "type": "input_image",
                "image_url": b64_data_uri(Path(img_path), dim=dim),
                "detail": detail,
            },
        ],
    }

    messages = [
        system_msg,
        developer_msg,
        *fewshot,
        user_msg,
    ]

    if dry_run:  # just print prompt length & return dummy
        print(f"[DRY-RUN] Would send prompt with {len(messages)} messages for {img_path}")
        with open(img_path.parent.parent / f"{img_path.stem}.json", "w") as f:
            json.dump(messages, f)
        return EstuaryResponse(label=MouthLabel.UNSURE, reasoning="Dry Run")

    resp = client.responses.parse(
        model=model,
        input=messages,
        text_format=EstuaryResponse,
    )
    if resp.output_parsed is None:
        print("None output_parsed")
        print(resp)
        return EstuaryResponse(label=MouthLabel.UNSURE, reasoning="Empty Msg")

    est_resp: EstuaryResponse = resp.output_parsed
    return est_resp


@click.command()
@click.option(
    "-d",
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Root directory for project with labels & prompts.",
)
@click.option(
    "-s",
    "--save-path",
    type=click.Path(exists=False, path_type=Path),
    required=True,
    help="Where to save the result",
)
@click.option(
    "--model", default="gpt-4.1-nano", show_default=True, help="Vision-capable model name"
)
@click.option("--detail", default="auto", show_default=True, help="Image detail level")
@click.option(
    "--dry-run", is_flag=True, help="Skip API calls and file writes; print what would happen"
)
@click.option("--limit", type=int, required=False, help="Limit the number of images per region")
@click.option("--dim", type=int, default=512, help="Image size")
def main(
    project_dir: Path,
    save_path: Path,
    model: str,
    detail: str,
    dry_run: bool,
    limit: int | None,
    dim: int,
) -> None:
    # Resolve labels.csv
    labels_path = project_dir / "labels.csv"
    cfg_path = Path("prompt.json")

    if not labels_path.exists():
        click.echo(f"Missing labels.csv: {labels_path}", err=True)
        raise SystemExit(1)
    if not cfg_path.exists():
        click.echo(f"Missing prompt.json: {cfg_path}", err=True)
        raise SystemExit(1)

    df = pd.read_csv(labels_path).sort_values(by=["region", "label", "source_jpeg"])
    prompt_cfg = load_prompt_cfg(cfg_path)

    # init client
    client = openai.Client()

    # Built region prompts once
    region_prompt_pths = defaultdict(dict)
    for (region, label), rdf in df.groupby(["region", "label"]):
        if label in ["open", "closed"]:
            region_prompt_pths[region][label] = [Path(rdf.iloc[0].source_jpeg)]
    region_prompts = {
        r: build_fewshot_messages(p, detail=detail, dim=dim) for r, p in region_prompt_pths.items()
    }

    results: list[
        tuple[str, str, str, str, str]
    ] = []  # region, source_jpeg, source_tif, pred, label

    for (region, label), rdf in tqdm.tqdm(df.groupby(["region", "label"])):
        if label in ["open", "closed"] and limit is not None:
            end = limit + 1
            rdf = rdf.iloc[1:end]
        elif label in ["open", "closed"]:
            rdf = rdf.iloc[1:]
        elif limit is not None:
            rdf = rdf.iloc[:limit]

        for _, row in rdf.iterrows():
            img_path = Path(row.source_jpeg)

            fewshot = region_prompts[region]
            resp = classify_image(
                client,
                model,
                fewshot,
                img_path,
                dry_run,
                prompt_cfg,
                detail=detail,
                dim=dim,
            )
            label = str(row["label"])
            results.append((region, row.source_jpeg, row.source_tif, resp.label, label))

    pred_df = pd.DataFrame(
        results, columns=["region", "source_jpeg", "source_tif", "pred", "label"]
    )
    if not dry_run:
        pred_df.to_csv(save_path)
        click.echo(f"Wrote predictions to {save_path}")
    else:
        click.echo("[DRY-RUN] Skipped predictions write.")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set the OPENAI_API_KEY environment variable.")
    main()
