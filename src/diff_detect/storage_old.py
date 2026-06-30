from __future__ import annotations

import base64
import io
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

DIFFERENCE_LABEL_STYLES: dict[DifferenceLabel, dict[str, str]] = {
    "shape": {"color": "#ffb000", "fill": "rgba(255, 176, 0, 0.2)"},
    "color": {"color": "#e83e8c", "fill": "rgba(232, 62, 140, 0.18)"},
    "texture": {"color": "#006d77", "fill": "rgba(0, 109, 119, 0.18)"},
}
DIFFERENCE_LABELS: tuple[DifferenceLabel, ...] = tuple(DIFFERENCE_LABEL_STYLES)


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


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    cleaned = value.lstrip("#")
    return (
        int(cleaned[0:2], 16),
        int(cleaned[2:4], 16),
        int(cleaned[4:6], 16),
    )


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
