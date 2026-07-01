from __future__ import annotations

import base64
import io
import os
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import BaseModel

from diff_detect.models import (
    ChallengeId,
    Dataset,
    ExplainDifferencesChallenge,
    Image,
    ImageKey,
    RatingEval,
    SelectionChoice,
    SelectionTask,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DEFAULT_DATASET_ID = "butterfly"
DEFAULT_CHALLENGE_ID: ChallengeId = "butterfly_easy"
DATASET_ID_ENV_VAR = "DIFF_DETECT_DATASET_ID"
CHALLENGE_ID_ENV_VAR = "DIFF_DETECT_CHALLENGE_ID"
CANVAS_WIDTH = 680
CANVAS_HEIGHT = 420

DifferenceLabel = Literal["shape", "color", "texture"]
WinnerSource = Literal["self", "peer", "ai"]


DIFFERENCE_LABEL_STYLES: dict[DifferenceLabel, dict[str, str]] = {
    "shape": {"color": "#ffb000", "fill": "rgba(255, 176, 0, 0.2)"},
    "color": {"color": "#e83e8c", "fill": "rgba(232, 62, 140, 0.18)"},
    "texture": {"color": "#006d77", "fill": "rgba(0, 109, 119, 0.18)"},
}
DIFFERENCE_LABELS: tuple[DifferenceLabel, ...] = tuple(DIFFERENCE_LABEL_STYLES)


@dataclass(frozen=True)
class ImageView:
    dataset_id: str
    image_id: str
    path: str
    source_url: str | None = None
    species_role: str = "reference"
    species: str = ""
    subspecies: str = ""
    view: str = ""
    mimic_group: str = ""
    hybrid_stat: str = ""
    source: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskMetadataView:
    dataset_id: str
    challenge_id: str
    round_rule: str = ""
    mimic_group: str = ""
    view: str = ""


@dataclass(frozen=True)
class RoundView:
    task_id: str
    images: tuple[ImageView, ...]
    metadata: TaskMetadataView
    difficulty: float = 0.5

    @property
    def odd_image_id(self) -> str:
        return self.images[-1].image_id


class RatingOption(BaseModel):
    option_id: str
    source: WinnerSource
    dataset_id: str
    task_id: str
    selected_image_id: str
    label: str
    explanation: str
    composite_png_base64: str
    submission_id: str | None = None


def configured_dataset_id(configured: str | None = None) -> str:
    value = os.environ.get(DATASET_ID_ENV_VAR) or configured or DEFAULT_DATASET_ID
    return normalize_dataset_id(value)


def configured_challenge_id(configured: str | None = None) -> ChallengeId:
    value = os.environ.get(CHALLENGE_ID_ENV_VAR) or configured or DEFAULT_CHALLENGE_ID
    if value not in ("dummy", "butterfly_easy", "butterfly_difficult"):
        raise ValueError(f"Unknown challenge id: {value}")
    return value


def normalize_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.strip()
    if not normalized:
        raise ValueError("Dataset id cannot be empty.")
    if normalized != Path(normalized).name or normalized in {".", ".."}:
        raise ValueError("Dataset id must be a directory name under data/.")
    return normalized


def dataset_root(dataset_id: str = DEFAULT_DATASET_ID) -> Path:
    return DATA_ROOT / normalize_dataset_id(dataset_id)


def dataset_path(dataset_id: str = DEFAULT_DATASET_ID) -> Path:
    return dataset_root(dataset_id) / f"{normalize_dataset_id(dataset_id)}.json"


def challenge_path(
    dataset_id: str = DEFAULT_DATASET_ID,
    challenge_id: ChallengeId = DEFAULT_CHALLENGE_ID,
) -> Path:
    return dataset_root(dataset_id) / f"{challenge_id}.json"


def available_dataset_ids(data_root: Path = DATA_ROOT) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and (path / f"{path.name}.json").is_file()
    )


def available_challenge_ids(dataset_id: str = DEFAULT_DATASET_ID) -> list[ChallengeId]:
    challenge_ids: list[ChallengeId] = []
    for value in ("dummy", "butterfly_easy", "butterfly_difficult"):
        challenge_id: ChallengeId = value
        if challenge_path(dataset_id, challenge_id).is_file():
            challenge_ids.append(challenge_id)
    return challenge_ids


@lru_cache(maxsize=16)
def load_dataset(dataset_id: str = DEFAULT_DATASET_ID) -> Dataset:
    return Dataset.model_validate_json(dataset_path(dataset_id).read_text())


@lru_cache(maxsize=32)
def load_selection_challenge(
    dataset_id: str = DEFAULT_DATASET_ID,
    challenge_id: ChallengeId = DEFAULT_CHALLENGE_ID,
) -> ExplainDifferencesChallenge:
    return ExplainDifferencesChallenge.model_validate_json(
        challenge_path(dataset_id, challenge_id).read_text()
    )


def load_rounds(
    dataset_id: str = DEFAULT_DATASET_ID,
    challenge_id: ChallengeId | None = None,
) -> list[RoundView]:
    active_challenge_id = configured_challenge_id(challenge_id)
    challenge = load_selection_challenge(dataset_id, active_challenge_id)
    image_by_key = dataset_image_index(dataset_id)
    return [
        round_view_from_task(
            task,
            dataset_id=dataset_id,
            challenge_id=active_challenge_id,
            task_index=index,
            image_by_key=image_by_key,
        )
        for index, task in enumerate(challenge.tasks)
    ]


def load_seeded_annotations(
    dataset_id: str = DEFAULT_DATASET_ID, path: Path | None = None
) -> list[Any]:
    return []


def dataset_image_index(
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict[tuple[str, str], Image]:
    images: dict[tuple[str, str], Image] = {}
    for image in load_dataset(dataset_id).root:
        images.setdefault((image.dataset_id, image.image_id), image)
    return images


def round_view_from_task(
    task: SelectionTask,
    *,
    dataset_id: str,
    challenge_id: str,
    task_index: int,
    image_by_key: dict[tuple[str, str], Image],
) -> RoundView:
    images = tuple(
        image_view_from_key(image_key, image_by_key) for image_key in task.images
    )
    return RoundView(
        task_id=task_id_for(challenge_id, task_index),
        images=images,
        metadata=TaskMetadataView(
            dataset_id=dataset_id,
            challenge_id=challenge_id,
            round_rule=challenge_id,
            mimic_group=shared_image_info(images, "mimic_group"),
            view=shared_image_info(images, "view"),
        ),
        difficulty=task.difficulty,
    )


def image_view_from_key(
    image_key: ImageKey, image_by_key: dict[tuple[str, str], Image]
) -> ImageView:
    image = image_by_key.get((image_key.dataset_id, image_key.image_id))
    if image:
        return image_view_from_image(image)
    return ImageView(
        dataset_id=image_key.dataset_id,
        image_id=image_key.image_id,
        path="",
    )


def image_view_from_image(image: Image) -> ImageView:
    info = image.image_info
    return ImageView(
        dataset_id=image.dataset_id,
        image_id=image.image_id,
        path=str(dataset_root(image.dataset_id) / image.path),
        source_url=info.get("file_url"),
        species=info.get("species", ""),
        subspecies=info.get("subspecies", ""),
        view=info.get("view", ""),
        mimic_group=info.get("mimic_group", ""),
        hybrid_stat=info.get("hybrid_stat", ""),
        source=dict(info),
    )


def shared_image_info(images: tuple[ImageView, ...], key: str) -> str:
    values = {str((image.source or {}).get(key, "")) for image in images}
    return values.pop() if len(values) == 1 else ""


def choose_rounds(
    rounds: list[RoundView], username: str, dataset_id: str | None = None
) -> list[RoundView]:
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


def task_id_for(challenge_id: str, task_index: int) -> str:
    return f"{challenge_id}:{task_index}"


def parse_task_id(task_id: str) -> tuple[str, int]:
    challenge_id, task_index = task_id.rsplit(":", 1)
    return challenge_id, int(task_index)


def image_for_id(task: RoundView, image_id: str) -> ImageView:
    for image in task.images:
        if image.image_id == image_id:
            return image
    raise KeyError(f"Image {image_id!r} is not part of task {task.task_id!r}.")


def reference_images(task: RoundView, selected_image_id: str) -> list[ImageView]:
    return [image for image in task.images if image.image_id != selected_image_id]


def load_image(
    image_spec: ImageView | Image | ImageKey | dict[str, Any],
    size: tuple[int, int] = (CANVAS_WIDTH, CANVAS_HEIGHT),
) -> PILImage.Image:
    image = coerce_image_view(image_spec)
    path = Path(image.path)
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        try:
            return PILImage.open(path).convert("RGB").resize(size)
        except Exception:
            path.unlink(missing_ok=True)
    if image.source_url:
        try:
            return load_remote_image(image.source_url, path, size)
        except Exception:
            pass
    return placeholder_image(image.image_id, size)


def coerce_image_view(
    image_spec: ImageView | Image | ImageKey | dict[str, Any],
) -> ImageView:
    if isinstance(image_spec, ImageView):
        return image_spec
    if isinstance(image_spec, Image):
        return image_view_from_image(image_spec)
    if isinstance(image_spec, ImageKey):
        return image_view_from_image(
            dataset_image_index(image_spec.dataset_id)[
                (image_spec.dataset_id, image_spec.image_id)
            ]
        )
    return ImageView(
        dataset_id=str(image_spec.get("dataset_id") or DEFAULT_DATASET_ID),
        image_id=str(image_spec["image_id"]),
        path=str(image_spec.get("path", "")),
        source_url=image_spec.get("source_url"),
        species_role=str(image_spec.get("species_role", "reference")),
    )


def load_remote_image(
    source_url: str, cache_path: Path, size: tuple[int, int]
) -> PILImage.Image:
    image_bytes = download_image_bytes(source_url)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(image_bytes)
    except OSError:
        pass
    return PILImage.open(io.BytesIO(image_bytes)).convert("RGB").resize(size)


@lru_cache(maxsize=128)
def download_image_bytes(source_url: str) -> bytes:
    quoted_url = urllib.parse.quote(source_url, safe=":/?&=%")
    request = urllib.request.Request(
        quoted_url,
        headers={"User-Agent": "specifly-streamlit/0.1"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        return response.read()


def placeholder_image(image_id: str, size: tuple[int, int]) -> PILImage.Image:
    width, height = size
    image = PILImage.new("RGBA", size, (248, 248, 244, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    cx, cy = width // 2, height // 2
    hue = sum(image_id.encode("utf-8")) % 255
    fill = (80 + hue // 4, 120 + hue // 6, 165 + hue // 8, 220)
    outline = (65, 58, 54, 255)

    left = [
        (cx - 24, cy),
        (width * 0.12, height * 0.18),
        (width * 0.08, height * 0.75),
        (cx - 42, height * 0.86),
    ]
    right = [(width - x, y) for x, y in left]
    body = (max(12, width // 38), max(70, height // 5))
    draw.polygon(left, fill=fill, outline=outline)
    draw.polygon(right, fill=fill, outline=outline)
    draw.ellipse((cx - body[0], cy - body[1], cx + body[0], cy + body[1]), fill=outline)
    return image


def encode_png(image: PILImage.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def decode_png(data: str) -> PILImage.Image:
    return PILImage.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")


def composite_annotation(background: PILImage.Image, overlay_data: Any) -> str:
    base = background.convert("RGBA").resize((CANVAS_WIDTH, CANVAS_HEIGHT))
    if overlay_data is None:
        return encode_png(base)
    overlay = PILImage.fromarray(np.asarray(overlay_data, dtype=np.uint8)).convert(
        "RGBA"
    )
    overlay = overlay.resize(base.size)
    return encode_png(PILImage.alpha_composite(base, overlay))


def canvas_has_objects(canvas_json: Any | None) -> bool:
    return bool(canvas_object_dicts(canvas_json))


def canvas_labels(
    canvas_json: Any | None,
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


def canvas_object_dicts(canvas_json: Any | None) -> list[dict[str, Any]]:
    if canvas_json is None:
        return []
    if hasattr(canvas_json, "objects"):
        return [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in canvas_json.objects
        ]
    objects = canvas_json.get("objects", [])
    return [item for item in objects if isinstance(item, dict)]


def label_display(labels: list[str] | str | None) -> str:
    if isinstance(labels, list):
        return ", ".join(labels)
    return labels or ""


def table_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", response)
    return data if isinstance(data, list) else []


def fetch_selection_choices(
    supabase: Any, username: str, dataset_id: str = DEFAULT_DATASET_ID
) -> list[SelectionChoice]:
    return [
        selection_choice_from_row(row)
        for row in fetch_selection_choice_rows(supabase, username, dataset_id)
    ]


def fetch_selection_choice_rows(
    supabase: Any, username: str, dataset_id: str = DEFAULT_DATASET_ID
) -> list[dict[str, Any]]:
    response = (
        supabase.table("selection_choices")
        .select("*")
        .eq("username", username)
        .eq("dataset_id", dataset_id)
        .execute()
    )
    return table_rows(response)


def fetch_user_submissions(
    supabase: Any, username: str, dataset_id: str = DEFAULT_DATASET_ID
) -> list[dict[str, Any]]:
    return [
        submission_row_from_selection_choice(row)
        for row in fetch_selection_choice_rows(supabase, username, dataset_id)
    ]


def fetch_rating_eval_rows(
    supabase: Any, username: str, dataset_id: str = DEFAULT_DATASET_ID
) -> list[dict[str, Any]]:
    response = (
        supabase.table("rating_evals")
        .select("*")
        .eq("username", username)
        .eq("dataset_id", dataset_id)
        .execute()
    )
    return table_rows(response)


def fetch_user_ratings(
    supabase: Any, username: str, dataset_id: str = DEFAULT_DATASET_ID
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "task_id": task_id_for(row["challenge_id"], int(row["task_index"])),
        }
        for row in fetch_rating_eval_rows(supabase, username, dataset_id)
    ]


def upsert_selection_choice(
    supabase: Any,
    selection_choice: SelectionChoice,
    *,
    dataset_id: str,
    task_id: str,
    selected_image_id: str,
    labels: list[DifferenceLabel],
    canvas_json: dict[str, Any],
    composite_png_base64: str,
) -> None:
    challenge_id, task_index = parse_task_id(task_id)
    row = {
        "dataset_id": dataset_id,
        "challenge_id": challenge_id,
        "task_index": task_index,
        "images": [image.model_dump(mode="json") for image in selection_choice.images],
        "username": selection_choice.user,
        "user_kind": selection_choice.user_kind,
        "choice_index": selection_choice.index,
        "explanation": selection_choice.explanation,
        "annotations": list(selection_choice.annotations),
        "artifacts": {
            "selected_image_id": selected_image_id,
            "labels": labels,
            "canvas_json": canvas_json,
            "composite_png_base64": composite_png_base64,
        },
    }
    supabase.table("selection_choices").upsert(
        row, on_conflict="dataset_id,challenge_id,task_index,username"
    ).execute()


def upsert_rating_eval(
    supabase: Any,
    rating_eval: RatingEval,
    *,
    dataset_id: str,
    task_id: str,
    options: list[RatingOption],
) -> None:
    challenge_id, task_index = parse_task_id(task_id)
    row = {
        "dataset_id": dataset_id,
        "challenge_id": challenge_id,
        "task_index": task_index,
        "username": rating_eval.user,
        "choices": [choice.model_dump(mode="json") for choice in rating_eval.choices],
        "most_convincing": rating_eval.most_convincing.model_dump(mode="json"),
        "most_likely_ai": (
            rating_eval.most_likely_ai.model_dump(mode="json")
            if rating_eval.most_likely_ai
            else None
        ),
        "artifacts": {
            "option_payload": [option.model_dump(mode="json") for option in options],
        },
    }
    supabase.table("rating_evals").upsert(
        row, on_conflict="dataset_id,challenge_id,task_index,username"
    ).execute()


def fetch_peer_submission(
    supabase: Any,
    username: str,
    task_id: str,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict[str, Any] | None:
    challenge_id, task_index = parse_task_id(task_id)
    response = (
        supabase.table("selection_choices")
        .select("*")
        .eq("dataset_id", dataset_id)
        .eq("challenge_id", challenge_id)
        .eq("task_index", task_index)
        .neq("username", username)
        .limit(1)
        .execute()
    )
    rows = table_rows(response)
    return submission_row_from_selection_choice(rows[0]) if rows else None


def selection_choice_from_row(row: dict[str, Any]) -> SelectionChoice:
    return SelectionChoice(
        images=tuple(ImageKey.model_validate(item) for item in row["images"]),
        user=row["username"],
        index=row["choice_index"],
        explanation=row.get("explanation"),
        user_kind=row["user_kind"],
        annotations=row.get("annotations") or [],
    )


def submission_row_from_selection_choice(row: dict[str, Any]) -> dict[str, Any]:
    artifacts = row.get("artifacts") or {}
    annotations = row.get("annotations") or []
    canvas_json = artifacts.get("canvas_json") or annotation_canvas_json(annotations)
    labels = artifacts.get("labels") or annotation_labels(annotations)
    selected_image_id = artifacts.get(
        "selected_image_id"
    ) or selected_image_id_from_row(row)
    task_id = task_id_for(row["challenge_id"], int(row["task_index"]))
    return {
        "id": row.get("id"),
        "username": row["username"],
        "dataset_id": row["dataset_id"],
        "task_id": task_id,
        "selected_image_id": selected_image_id,
        "labels": labels,
        "explanation": row.get("explanation") or "",
        "canvas_json": canvas_json or {},
        "annotation_layers": annotations[0] if annotations else {},
        "composite_png_base64": artifacts.get("composite_png_base64") or "",
    }


def selected_image_id_from_row(row: dict[str, Any]) -> str:
    images = row.get("images") or []
    choice_index = int(row["choice_index"])
    if 0 <= choice_index < len(images):
        return str(images[choice_index]["image_id"])
    return ""


def annotation_canvas_json(annotations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if annotations:
        canvas_json = annotations[0].get("canvas_json")
        if isinstance(canvas_json, dict):
            return canvas_json
    return None


def annotation_labels(annotations: list[dict[str, Any]]) -> list[str]:
    if annotations:
        labels = annotations[0].get("labels")
        if isinstance(labels, list):
            return [str(label) for label in labels]
    return []


def build_rating_options(
    task: RoundView,
    own_submission: dict[str, Any],
    peer_submission: dict[str, Any] | None,
    seeded_annotations: list[Any],
    username: str,
    dataset_id: str | None = None,
) -> list[RatingOption]:
    task_id = task.task_id
    options = [
        RatingOption(
            option_id=f"self:{task_id}",
            source="self",
            dataset_id=own_submission.get("dataset_id")
            or dataset_id
            or DEFAULT_DATASET_ID,
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
                dataset_id=peer_submission.get("dataset_id")
                or dataset_id
                or DEFAULT_DATASET_ID,
                task_id=task_id,
                selected_image_id=peer_submission["selected_image_id"],
                label=label_display(peer_submission["labels"]),
                explanation=peer_submission.get("explanation") or "",
                composite_png_base64=peer_submission["composite_png_base64"],
                submission_id=str(peer_submission.get("id") or ""),
            )
        )

    random.Random(f"{username}:{task_id}:ratings").shuffle(options)
    return options
