from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from diff_detect.challenges import DATA_DIR
from diff_detect.common import DIFFERENCE_LABEL_STYLES, EXPLAIN_CANVAS_SCALE
from diff_detect.models import (
    Annotation,
    DatasetId,
    ExplainOutcome,
    ImageId,
    User,
    UserKind,
    UserRole,
)

TRIPLE_ID_PATTERN = re.compile(r"triple_\d{4}")
AI_USER_ID = "ai"


def _triple_id(value: str) -> str:
    match = TRIPLE_ID_PATTERN.search(value)
    if match is None:
        raise ValueError(f"Could not find triple id in {value!r}.")
    return match.group(0)


def _path_keys(path: str) -> set[str]:
    path_obj = Path(path)
    return {
        path_obj.as_posix(),
        (Path(path_obj.parent.name) / path_obj.name).as_posix(),
    }


def _read_image_id_by_path(download_dir: Path) -> dict[str, ImageId]:
    image_id_by_path: dict[str, ImageId] = {}
    for index_path in [download_dir / "easy.csv", download_dir / "difficult.csv"]:
        with index_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_id = ImageId(Path(row["output_path"]).stem)
                for key in _path_keys(row["output_path"]):
                    image_id_by_path[key] = image_id
    return image_id_by_path


def _iter_diagnostic_records(download_dir: Path) -> Iterable[dict[str, Any]]:
    for difficulty in ["easy", "difficult"]:
        for path in sorted((download_dir / difficulty).glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)

            records = payload.get("records") if isinstance(payload, dict) else None
            if records is None:
                records = payload.values()
            yield from records


def _read_diagnostic_records_by_triple(download_dir: Path) -> dict[str, dict[str, Any]]:
    records_by_triple: dict[str, dict[str, Any]] = {}
    for record in _iter_diagnostic_records(download_dir):
        records_by_triple[_triple_id(record["image_id"])] = record
    return records_by_triple


def _iter_separate_box_paths(download_dir: Path) -> Iterable[Path]:
    for subdir in [
        download_dir / "easy" / "separate_bounding_box",
        download_dir / "difficult" / "separate_bounding_boxes",
    ]:
        yield from sorted(subdir.glob("*.json"))


def _subimage_image_ids(
    subimages: dict[str, Any], image_id_by_path: dict[str, ImageId]
) -> dict[str, ImageId]:
    image_ids: dict[str, ImageId] = {}
    for label, subimage in subimages.items():
        for key in _path_keys(subimage["image_path"]):
            image_id = image_id_by_path.get(key)
            if image_id is not None:
                image_ids[label] = image_id
                break
        else:
            raise KeyError(f"No CSV image id found for {subimage['image_path']!r}.")
    return image_ids


def _odd_subimage_label(subimages: dict[str, Any], unique_specimen: str) -> str:
    for label, subimage in subimages.items():
        if any(box.get("specimen") == unique_specimen for box in subimage["boxes"]):
            return label
    raise ValueError(f"No subimage contains boxes for {unique_specimen!r}.")


def _fabric_rect_from_box(box: dict[str, Any]) -> dict[str, Any]:
    x_min, y_min, x_max, y_max = box["bbox_local_pixels"]
    left = x_min * EXPLAIN_CANVAS_SCALE
    top = y_min * EXPLAIN_CANVAS_SCALE
    width = (x_max - x_min) * EXPLAIN_CANVAS_SCALE
    height = (y_max - y_min) * EXPLAIN_CANVAS_SCALE
    stroke = DIFFERENCE_LABEL_STYLES["color"]["color"]
    return {
        "type": "rect",
        "version": "4.4.0",
        "originX": "left",
        "originY": "top",
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "fill": "rgba(0, 0, 0, 0)",
        "stroke": stroke,
        "strokeWidth": 3,
        "strokeDashArray": None,
        "strokeLineCap": "butt",
        "strokeDashOffset": 0,
        "strokeLineJoin": "miter",
        "strokeUniform": False,
        "strokeMiterLimit": 4,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "flipX": False,
        "flipY": False,
        "opacity": 1,
        "shadow": None,
        "visible": True,
        "backgroundColor": "",
        "fillRule": "nonzero",
        "paintFirst": "fill",
        "globalCompositeOperation": "source-over",
        "skewX": 0,
        "skewY": 0,
        "rx": 0,
        "ry": 0,
        "ai_feature": box.get("feature"),
        "ai_short_label": box.get("short_label"),
        "ai_wing_slot": box.get("wing_slot"),
        "ai_label": box.get("label"),
        "ai_importance": box.get("importance"),
    }


def _annotation_from_boxes(boxes: list[dict[str, Any]]) -> Annotation:
    return Annotation.model_validate(
        {
            "raw": {
                "version": "4.4.0",
                "objects": [_fabric_rect_from_box(box) for box in boxes],
            }
        }
    )


def iter_ai_annotation_outcomes(
    download_dir: Path | None = None, *, user_id: str = AI_USER_ID
) -> Iterable[ExplainOutcome]:
    if download_dir is None:
        download_dir = DATA_DIR / DatasetId.BUTTERFLY / "download"

    image_id_by_path = _read_image_id_by_path(download_dir)
    records_by_triple = _read_diagnostic_records_by_triple(download_dir)

    for path in _iter_separate_box_paths(download_dir):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        triple_id = _triple_id(path.name)
        record = records_by_triple[triple_id]
        subimages = payload["subimages"]
        image_ids_by_subimage = _subimage_image_ids(subimages, image_id_by_path)
        odd_label = _odd_subimage_label(subimages, record["unique_species_candidate"])
        annotated_image = image_ids_by_subimage[odd_label]
        reference_images = [
            image_id
            for label, image_id in image_ids_by_subimage.items()
            if label != odd_label
        ]
        if len(reference_images) != 2:
            raise ValueError(f"Expected two reference images for {triple_id}.")

        yield ExplainOutcome(
            dataset_id=DatasetId.BUTTERFLY,
            annotated_image=annotated_image,
            reference_image1=reference_images[0],
            reference_image2=reference_images[1],
            user=user_id,
            explanation=record.get("basis") or "AI bounding-box solution.",
            annotation=_annotation_from_boxes(subimages[odd_label]["boxes"]),
        )


def ensure_ai_user(storage: Any, *, user_id: str = AI_USER_ID) -> None:
    if storage.fetch_user(user_id) is not None:
        return
    storage.add_user(
        User(
            id=user_id,
            name="AI",
            lab=None,
            kind=UserKind.AI,
            role=UserRole.PARTICIPANT,
            hashed_password=None,
        )
    )


def import_ai_annotations(
    storage: Any, download_dir: Path | None = None, *, user_id: str = AI_USER_ID
) -> list[ExplainOutcome]:
    ensure_ai_user(storage, user_id=user_id)
    outcomes = list(iter_ai_annotation_outcomes(download_dir, user_id=user_id))
    for outcome in outcomes:
        storage.upsert_explain_outcome(outcome)
    return outcomes
