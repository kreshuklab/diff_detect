from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
ROUND_MANIFEST = ROOT / "data" / "rounds.json"
SEEDED_ANNOTATIONS = ROOT / "data" / "seeded_annotations.json"
CANVAS_WIDTH = 680
CANVAS_HEIGHT = 420
MIN_ROUNDS = 3


LABEL_STYLES = {
    "shape": {"color": "#111111", "fill": "rgba(17, 17, 17, 0.18)"},
    "color": {"color": "#e83e8c", "fill": "rgba(232, 62, 140, 0.18)"},
    "texture": {"color": "#006d77", "fill": "rgba(0, 109, 119, 0.18)"},
}
LABELS = tuple(LABEL_STYLES)


@dataclass(frozen=True)
class RatingOption:
    option_id: str
    source: str
    task_id: str
    selected_image_id: str
    label: str
    explanation: str
    composite_png_base64: str
    submission_id: str | None = None


def load_rounds(path: Path = ROUND_MANIFEST) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_seeded_annotations(path: Path = SEEDED_ANNOTATIONS) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_rounds(rounds: list[dict[str, Any]], username: str, limit: int = MIN_ROUNDS) -> list[dict[str, Any]]:
    selected = list(rounds)
    random.Random(username).shuffle(selected)
    return selected[: min(limit, len(selected))]


def image_for_id(task: dict[str, Any], image_id: str) -> dict[str, Any]:
    for image in task["images"]:
        if image["image_id"] == image_id:
            return image
    raise KeyError(f"Image {image_id!r} is not part of task {task['task_id']!r}.")


def reference_images(task: dict[str, Any], selected_image_id: str) -> list[dict[str, Any]]:
    return [image for image in task["images"] if image["image_id"] != selected_image_id]


def load_wing_image(image_spec: dict[str, Any], size: tuple[int, int] = (CANVAS_WIDTH, CANVAS_HEIGHT)) -> Image.Image:
    path = ROOT / image_spec["path"]
    if path.exists():
        try:
            return Image.open(path).convert("RGB").resize(size)
        except Exception:
            path.unlink(missing_ok=True)
    source_url = image_spec.get("source_url")
    if source_url:
        try:
            return load_remote_wing_image(source_url, path, size)
        except Exception:
            pass
    return placeholder_wing_image(image_spec["image_id"], image_spec.get("species_role", "reference"), size)


def load_remote_wing_image(source_url: str, cache_path: Path, size: tuple[int, int]) -> Image.Image:
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


def placeholder_wing_image(image_id: str, species_role: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    base_hue = digest[0]
    accent_hue = digest[1]
    if species_role == "odd":
        accent_hue = (accent_hue + 90) % 255

    bg = Image.new("RGBA", size, (248, 248, 244, 255))
    draw = ImageDraw.Draw(bg, "RGBA")
    cx, cy = width // 2, height // 2

    left = [
        (cx - 24, cy),
        (width * 0.12, height * 0.18),
        (width * 0.08, height * 0.75),
        (cx - 42, height * 0.86),
    ]
    right = [(width - x, y) for x, y in left]
    body = (max(12, width // 38), max(70, height // 5))
    main = (80 + base_hue // 3, 70 + accent_hue // 4, 150 + base_hue // 5, 210)
    accent = (190 + accent_hue // 6, 90 + base_hue // 5, 80 + accent_hue // 7, 170)
    outline = (65, 58, 54, 255)

    draw.polygon(left, fill=main, outline=outline)
    draw.polygon(right, fill=main, outline=outline)
    draw.ellipse((cx - body[0], cy - body[1], cx + body[0], cy + body[1]), fill=outline)
    for offset in (-78, -36, 36, 78):
        draw.arc((cx - 220, cy - 155 + offset, cx - 8, cy + 120 + offset), 205, 326, fill=accent, width=5)
        draw.arc((cx + 8, cy - 155 + offset, cx + 220, cy + 120 + offset), 214, 335, fill=accent, width=5)

    if species_role == "odd":
        draw.ellipse((cx - 210, cy - 54, cx - 120, cy + 36), outline=(232, 62, 140, 210), width=8)
        draw.ellipse((cx + 120, cy - 54, cx + 210, cy + 36), outline=(232, 62, 140, 210), width=8)

    return bg


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


def seeded_rating_option(seed: dict[str, Any], task: dict[str, Any]) -> RatingOption:
    selected = image_for_id(task, seed["selected_image_id"])
    background = load_wing_image(selected)
    color = seed.get("annotation_color", LABEL_STYLES.get(seed["label"], LABEL_STYLES["shape"])["color"])
    composite = synthetic_annotation(background, color, seed["label"])
    return RatingOption(
        option_id=f"{seed['source']}:{seed['task_id']}:{seed['label']}",
        source=seed["source"],
        task_id=seed["task_id"],
        selected_image_id=seed["selected_image_id"],
        label=seed["label"],
        explanation=seed.get("explanation", ""),
        composite_png_base64=composite,
    )


def synthetic_annotation(background: Image.Image, color: str, label: str) -> str:
    image = background.convert("RGBA").resize((CANVAS_WIDTH, CANVAS_HEIGHT))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rgb = hex_to_rgb(color)
    if label == "shape":
        draw.rounded_rectangle((88, 68, CANVAS_WIDTH - 88, CANVAS_HEIGHT - 58), radius=36, outline=rgb + (230,), width=10)
    elif label == "color":
        for index, stripe in enumerate(["#f94144", "#f8961e", "#f9c74f", "#43aa8b", "#577590"]):
            stripe_rgb = hex_to_rgb(stripe)
            draw.arc((115 + index * 8, 92 + index * 7, CANVAS_WIDTH - 115, CANVAS_HEIGHT - 82), 200, 336, fill=stripe_rgb + (220,), width=6)
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


def canvas_has_objects(canvas_json: dict[str, Any] | None) -> bool:
    return bool(canvas_json and canvas_json.get("objects"))


def canvas_labels(canvas_json: dict[str, Any] | None, fallback_label: str | None = None) -> list[str]:
    if not canvas_json:
        return [fallback_label] if fallback_label else []

    label_by_color = {
        style["color"].lower(): label
        for label, style in LABEL_STYLES.items()
    }
    labels = []
    for item in canvas_json.get("objects", []):
        if not isinstance(item, dict):
            continue
        stroke = str(item.get("stroke", "")).lower()
        label = label_by_color.get(stroke)
        if label and label not in labels:
            labels.append(label)

    if not labels and fallback_label and canvas_has_objects(canvas_json):
        labels.append(fallback_label)
    return labels


def label_display(labels: list[str] | str | None) -> str:
    if isinstance(labels, list):
        return ", ".join(labels)
    return labels or ""


def table_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", response)
    return data if isinstance(data, list) else []


def fetch_user_submissions(supabase: Any, username: str) -> list[dict[str, Any]]:
    response = supabase.table("submissions").select("*").eq("username", username).execute()
    return table_rows(response)


def fetch_user_ratings(supabase: Any, username: str) -> list[dict[str, Any]]:
    response = supabase.table("ratings").select("*").eq("username", username).execute()
    return table_rows(response)


def upsert_submission(supabase: Any, payload: dict[str, Any]) -> None:
    supabase.table("submissions").upsert(payload, on_conflict="username,task_id").execute()


def upsert_rating(supabase: Any, payload: dict[str, Any]) -> None:
    supabase.table("ratings").upsert(payload, on_conflict="username,task_id").execute()


def fetch_peer_submission(supabase: Any, username: str, task_id: str) -> dict[str, Any] | None:
    response = (
        supabase.table("submissions")
        .select("*")
        .eq("task_id", task_id)
        .neq("username", username)
        .limit(1)
        .execute()
    )
    rows = table_rows(response)
    return rows[0] if rows else None


def build_rating_options(
    task: dict[str, Any],
    own_submission: dict[str, Any],
    peer_submission: dict[str, Any] | None,
    seeded_annotations: list[dict[str, Any]],
    username: str,
) -> list[RatingOption]:
    task_id = task["task_id"]
    seeds = [seed for seed in seeded_annotations if seed["task_id"] == task_id]
    ai_seed = next((seed for seed in seeds if seed["source"] == "ai"), None)
    peer_seed = next((seed for seed in seeds if seed["source"] == "peer"), None)

    options = [
        RatingOption(
            option_id=f"self:{task_id}",
            source="self",
            task_id=task_id,
            selected_image_id=own_submission["selected_image_id"],
            label=label_display(own_submission.get("labels") or own_submission["label"]),
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
                task_id=task_id,
                selected_image_id=peer_submission["selected_image_id"],
                label=label_display(peer_submission.get("labels") or peer_submission["label"]),
                explanation=peer_submission.get("explanation") or "",
                composite_png_base64=peer_submission["composite_png_base64"],
                submission_id=str(peer_submission.get("id") or ""),
            )
        )
    elif peer_seed:
        options.append(seeded_rating_option(peer_seed, task))

    if ai_seed:
        options.append(seeded_rating_option(ai_seed, task))

    random.Random(f"{username}:{task_id}:ratings").shuffle(options)
    return options
