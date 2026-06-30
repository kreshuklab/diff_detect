from __future__ import annotations

import base64
import io
import json
import os
import random
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DEFAULT_DATASET_ID = "hf_heliconius"
DATASET_ID_ENV_VAR = "DIFF_DETECT_DATASET_ID"
ROUND_MANIFEST_FILENAME = "rounds.json"
SEEDED_ANNOTATIONS_FILENAME = "seeded_annotations.json"
CANVAS_WIDTH = 680
CANVAS_HEIGHT = 420


DIFFERENCE_LABEL_STYLES: dict[DifferenceLabel, dict[str, str]] = {
    "shape": {"color": "#ffb000", "fill": "rgba(255, 176, 0, 0.2)"},
    "color": {"color": "#e83e8c", "fill": "rgba(232, 62, 140, 0.18)"},
    "texture": {"color": "#006d77", "fill": "rgba(0, 109, 119, 0.18)"},
}
DIFFERENCE_LABELS: tuple[DifferenceLabel, ...] = tuple(DIFFERENCE_LABEL_STYLES)


ROUND_LIST_ADAPTER = TypeAdapter(list[Round])
SEEDED_ANNOTATION_LIST_ADAPTER = TypeAdapter(list[SeededAnnotation])


def configured_dataset_id(configured: str | None = None) -> str:
    value = os.environ.get(DATASET_ID_ENV_VAR) or configured or DEFAULT_DATASET_ID
    return normalize_dataset_id(value)


def normalize_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.strip()
    if not normalized:
        raise ValueError("Dataset id cannot be empty.")
    if normalized != Path(normalized).name or normalized in {".", ".."}:
        raise ValueError("Dataset id must be a directory name under data/.")
    return normalized


def dataset_root(dataset_id: str = DEFAULT_DATASET_ID) -> Path:
    return DATA_ROOT / normalize_dataset_id(dataset_id)


def round_manifest_path(dataset_id: str = DEFAULT_DATASET_ID) -> Path:
    return dataset_root(dataset_id) / ROUND_MANIFEST_FILENAME


def seeded_annotations_path(dataset_id: str = DEFAULT_DATASET_ID) -> Path:
    return dataset_root(dataset_id) / SEEDED_ANNOTATIONS_FILENAME


def available_dataset_ids(data_root: Path = DATA_ROOT) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and (path / ROUND_MANIFEST_FILENAME).is_file()
    )


def load_rounds(
    dataset_id: str = DEFAULT_DATASET_ID, path: Path | None = None
) -> list[Round]:
    manifest_path = path or round_manifest_path(dataset_id)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return ROUND_LIST_ADAPTER.validate_python(json.load(handle))


def load_seeded_annotations(
    dataset_id: str = DEFAULT_DATASET_ID, path: Path | None = None
) -> list[SeededAnnotation]:
    annotations_path = path or seeded_annotations_path(dataset_id)
    if not annotations_path.exists():
        return []
    with annotations_path.open("r", encoding="utf-8") as handle:
        return SEEDED_ANNOTATION_LIST_ADAPTER.validate_python(json.load(handle))


def choose_rounds(
    rounds: list[Round], username: str, dataset_id: str | None = None
) -> list[Round]:
    selected = list(rounds)
    seed = f"{dataset_id}:{username}" if dataset_id else username
    random.Random(seed).shuffle(selected)
    return selected


def completed_task_ids(
    rows: list[dict[str, Any]], valid_task_ids: set[str]
) -> set[str]:
    return {
        task_id
        for row in rows
        if isinstance((task_id := row.get("task_id")), str)
        and task_id in valid_task_ids
    }


def image_for_id(task: Round, image_id: str) -> RoundImage:
    for image in task.images:
        if image.image_id == image_id:
            return image
    raise KeyError(f"Image {image_id!r} is not part of task {task.task_id!r}.")


def reference_images(task: Round, selected_image_id: str) -> list[RoundImage]:
    return [image for image in task.images if image.image_id != selected_image_id]


def load_image(
    image_spec: RoundImage | dict[str, Any],
    size: tuple[int, int] = (CANVAS_WIDTH, CANVAS_HEIGHT),
) -> Image.Image:
    image = (
        image_spec
        if isinstance(image_spec, RoundImage)
        else RoundImage.model_validate(image_spec)
    )
    path = ROOT / image.path
    if path.exists():
        try:
            return Image.open(path).convert("RGB").resize(size)
        except Exception:
            path.unlink(missing_ok=True)
    if image.source_url:
        try:
            return load_remote_image(image.source_url, path, size)
        except Exception:
            pass
    return placeholder_image(image.image_id, image.species_role, size)


def load_remote_image(
    source_url: str, cache_path: Path, size: tuple[int, int]
) -> Image.Image:
    image_bytes = download_image_bytes(source_url)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(image_bytes)
    except OSError:
        pass
    return Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(size)


@lru_cache(maxsize=128)
def download_image_bytes(source_url: str) -> bytes:
    quoted_url = urllib.parse.quote(source_url, safe=":/?&=%")
    request = urllib.request.Request(
        quoted_url,
        headers={"User-Agent": "specifly-streamlit/0.1"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        return response.read()


def encode_png(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def decode_png(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")


def composite_annotation(background: Image.Image, overlay_data: Any) -> str:
    base = background.convert("RGBA").resize((CANVAS_WIDTH, CANVAS_HEIGHT))
    if overlay_data is None:
        return encode_png(base)
    overlay = Image.fromarray(np.asarray(overlay_data, dtype=np.uint8)).convert("RGBA")
    overlay = overlay.resize(base.size)
    return encode_png(Image.alpha_composite(base, overlay))


def seeded_rating_option(
    seed: SeededAnnotation, task: Round, dataset_id: str | None = None
) -> RatingOption:
    selected = image_for_id(task, seed.selected_image_id)
    background = load_image(selected)
    color = seed.annotation_color or DIFFERENCE_LABEL_STYLES[seed.label]["color"]
    composite = synthetic_annotation(background, color, seed.label)
    return RatingOption(
        option_id=f"{seed.source}:{seed.task_id}:{seed.label}",
        source=seed.source,
        dataset_id=task.metadata.dataset_id or dataset_id or DEFAULT_DATASET_ID,
        task_id=seed.task_id,
        selected_image_id=seed.selected_image_id,
        label=seed.label,
        explanation=seed.explanation,
        composite_png_base64=composite,
    )


def synthetic_annotation(background: Image.Image, color: str, label: str) -> str:
    image = background.convert("RGBA").resize((CANVAS_WIDTH, CANVAS_HEIGHT))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rgb = hex_to_rgb(color)
    if label == "shape":
        draw.rounded_rectangle(
            (88, 68, CANVAS_WIDTH - 88, CANVAS_HEIGHT - 58),
            radius=36,
            outline=rgb + (230,),
            width=10,
        )
    elif label == "color":
        for index, stripe in enumerate(
            ["#f94144", "#f8961e", "#f9c74f", "#43aa8b", "#577590"]
        ):
            stripe_rgb = hex_to_rgb(stripe)
            draw.arc(
                (
                    115 + index * 8,
                    92 + index * 7,
                    CANVAS_WIDTH - 115,
                    CANVAS_HEIGHT - 82,
                ),
                200,
                336,
                fill=stripe_rgb + (220,),
                width=6,
            )
    else:
        for y in range(96, CANVAS_HEIGHT - 80, 34):
            draw.line((150, y, CANVAS_WIDTH - 150, y + 18), fill=rgb + (210,), width=6)
    return encode_png(Image.alpha_composite(image, overlay))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    cleaned = value.lstrip("#")
    return (
        int(cleaned[0:2], 16),
        int(cleaned[2:4], 16),
        int(cleaned[4:6], 16),
    )


def canvas_has_objects(canvas_json: CanvasJson | dict[str, Any] | None) -> bool:
    return bool(canvas_object_dicts(canvas_json))


def canvas_labels(
    canvas_json: CanvasJson | dict[str, Any] | None,
    fallback_label: DifferenceLabel | None = None,
) -> list[DifferenceLabel]:
    if not canvas_json:
        return [fallback_label] if fallback_label else []

    label_by_color: dict[str, DifferenceLabel] = {
        style["color"].lower(): label
        for label, style in DIFFERENCE_LABEL_STYLES.items()
    }
    labels: list[DifferenceLabel] = []
    for item in canvas_object_dicts(canvas_json):
        stroke = str(item.get("stroke", "")).lower()
        label = label_by_color.get(stroke)
        if label and label not in labels:
            labels.append(label)

    if not labels and fallback_label and canvas_has_objects(canvas_json):
        labels.append(fallback_label)
    return labels


def canvas_object_dicts(
    canvas_json: CanvasJson | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if canvas_json is None:
        return []
    if isinstance(canvas_json, CanvasJson):
        return [item.model_dump(mode="json") for item in canvas_json.objects]

    objects = canvas_json.get("objects", [])
    return [item for item in objects if isinstance(item, dict)]


def label_display(labels: list[str] | str | None) -> str:
    if isinstance(labels, list):
        return ", ".join(labels)
    return labels or ""


def table_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", response)
    return data if isinstance(data, list) else []


def fetch_user_submissions(
    supabase: Any, username: str, dataset_id: str = DEFAULT_DATASET_ID
) -> list[dict[str, Any]]:
    response = (
        supabase.table("submissions")
        .select("*")
        .eq("username", username)
        .eq("dataset_id", dataset_id)
        .execute()
    )
    return table_rows(response)


def fetch_user_ratings(
    supabase: Any, username: str, dataset_id: str = DEFAULT_DATASET_ID
) -> list[dict[str, Any]]:
    response = (
        supabase.table("ratings")
        .select("*")
        .eq("username", username)
        .eq("dataset_id", dataset_id)
        .execute()
    )
    return table_rows(response)


def upsert_submission(supabase: Any, payload: SubmissionPayload) -> None:
    supabase.table("submissions").upsert(
        payload.model_dump(mode="json"), on_conflict="username,dataset_id,task_id"
    ).execute()


def upsert_rating(supabase: Any, payload: RatingPayload) -> None:
    supabase.table("ratings").upsert(
        payload.model_dump(mode="json"), on_conflict="username,dataset_id,task_id"
    ).execute()


def fetch_peer_submission(
    supabase: Any,
    username: str,
    task_id: str,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict[str, Any] | None:
    response = (
        supabase.table("submissions")
        .select("*")
        .eq("task_id", task_id)
        .eq("dataset_id", dataset_id)
        .neq("username", username)
        .limit(1)
        .execute()
    )
    rows = table_rows(response)
    return rows[0] if rows else None


def build_rating_options(
    task: Round,
    own_submission: dict[str, Any],
    peer_submission: dict[str, Any] | None,
    seeded_annotations: list[SeededAnnotation],
    username: str,
    dataset_id: str | None = None,
) -> list[RatingOption]:
    task_id = task.task_id
    seeds = [seed for seed in seeded_annotations if seed.task_id == task_id]
    ai_seed = next((seed for seed in seeds if seed.source == "ai"), None)
    peer_seed = next((seed for seed in seeds if seed.source == "peer"), None)

    options = [
        RatingOption(
            option_id=f"self:{task_id}",
            source="self",
            dataset_id=(
                own_submission.get("dataset_id")
                or task.metadata.dataset_id
                or dataset_id
                or DEFAULT_DATASET_ID
            ),
            task_id=task_id,
            selected_image_id=own_submission["selected_image_id"],
            label=label_display(own_submission["labels"]),
            explanation=own_submission.get("explanation") or "",
            composite_png_base64=own_submission["composite_png_base64"],
            submission_id=str(own_submission.get("id") or f"{username}:{task_id}"),
        )
    ]

    if peer_submission:
        options.append(
            RatingOption(
                option_id=f"peer:{peer_submission.get('id', task_id)}",
                source="peer",
                dataset_id=(
                    peer_submission.get("dataset_id")
                    or task.metadata.dataset_id
                    or dataset_id
                    or DEFAULT_DATASET_ID
                ),
                task_id=task_id,
                selected_image_id=peer_submission["selected_image_id"],
                label=label_display(peer_submission["labels"]),
                explanation=peer_submission.get("explanation") or "",
                composite_png_base64=peer_submission["composite_png_base64"],
                submission_id=str(peer_submission.get("id") or ""),
            )
        )
    elif peer_seed:
        options.append(seeded_rating_option(peer_seed, task, dataset_id))

    if ai_seed:
        options.append(seeded_rating_option(ai_seed, task, dataset_id))

    random.Random(f"{username}:{task_id}:ratings").shuffle(options)
    return options
