from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image as PILImage

from diff_detect.challenges import DATA_DIR
from diff_detect.common import (
    DIFFERENCE_LABEL_STYLES,
    EXPLAIN_CANVAS_SCALE,
    EXPLAIN_STROKE_WIDTH,
)
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
FLYBUTTER_BATCH_FILENAME = (
    "butterfly_diagnostic_features_batch_0001_0002_0003_0004_0005.json"
)
FLYBUTTER_POSITIONS = (
    "specimen_1_left",
    "specimen_2_middle",
    "specimen_3_right",
)


def _ai_choice_user_id(user_id: str, subimage_label: str) -> str:
    return f"{user_id}_{subimage_label}"


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
    for index_path in sorted(download_dir.glob("*.csv")):
        with index_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                path = row["path"]
                image_id = ImageId(Path(path).stem)
                for key in _path_keys(path):
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
        "strokeWidth": EXPLAIN_STROKE_WIDTH,
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


def _dataset_id_from_download_dir(download_dir: Path) -> DatasetId:
    if (download_dir / FLYBUTTER_BATCH_FILENAME).exists():
        return DatasetId.FLYBUTTER
    try:
        return DatasetId(download_dir.parent.name)
    except ValueError:
        return DatasetId.BUTTERFLY


def _iter_butterfly_annotation_outcomes(
    download_dir: Path, *, user_id: str, dataset_id: DatasetId
) -> Iterable[ExplainOutcome]:
    image_id_by_path = _read_image_id_by_path(download_dir)
    records_by_triple = _read_diagnostic_records_by_triple(download_dir)

    for path in _iter_separate_box_paths(download_dir):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        triple_id = _triple_id(path.name)
        record = records_by_triple[triple_id]
        subimages = payload["subimages"]
        image_ids_by_subimage = _subimage_image_ids(subimages, image_id_by_path)
        _odd_subimage_label(subimages, record["unique_species_candidate"])
        for selected_label in sorted(subimages):
            annotated_image = image_ids_by_subimage[selected_label]
            reference_images = [
                image_ids_by_subimage[label]
                for label in sorted(image_ids_by_subimage)
                if label != selected_label
            ]
            if len(reference_images) != 2:
                raise ValueError(f"Expected two reference images for {triple_id}.")

            yield ExplainOutcome(
                dataset_id=dataset_id,
                annotated_image=annotated_image,
                reference_image1=reference_images[0],
                reference_image2=reference_images[1],
                user=_ai_choice_user_id(user_id, selected_label),
                explanation=record.get("basis") or "AI bounding-box solution.",
                annotation=_annotation_from_boxes(subimages[selected_label]["boxes"]),
            )


def _flybutter_specimen_info(
    download_dir: Path,
    triple_id: str,
    record: dict[str, Any],
    image_id_by_path: dict[str, ImageId],
) -> dict[str, dict[str, Any]]:
    concat_path = download_dir / triple_id / f"{triple_id}_segmented_concat.png"
    if not concat_path.exists():
        concat_path = download_dir / triple_id / record["source_file"]
    concat_width, _ = PILImage.open(concat_path).size
    json_size = record["image_size_px"]
    json_width = json_size["width"]
    json_height = json_size["height"]

    rows = []
    with (download_dir / triple_id / f"{triple_id}_concat_order.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    image_widths = []
    for row in rows:
        image_path = download_dir / Path(row["labeled_output_path"]).parent.name
        image_path = image_path / Path(row["labeled_output_path"]).name
        width, height = PILImage.open(image_path).size
        image_widths.append((width, height))

    gap = (concat_width - sum(width for width, _ in image_widths)) / (len(rows) - 1)
    offset = 0.0
    info: dict[str, dict[str, Any]] = {}
    for position, row, (width, height) in zip(FLYBUTTER_POSITIONS, rows, image_widths):
        image_id = None
        for key in _path_keys(row["labeled_output_path"]):
            image_id = image_id_by_path.get(key)
            if image_id is not None:
                break
        if image_id is None:
            raise KeyError(f"No CSV image id found for {row['labeled_output_path']!r}.")

        info[position] = {
            "label": row["label"],
            "image_id": image_id,
            "x_offset": offset * json_width / concat_width,
            "x_scale": width / (width * json_width / concat_width),
            "y_scale": height / json_height,
        }
        offset += width + gap
    return info


def _flybutter_local_box(
    bbox: list[float], specimen: dict[str, Any]
) -> list[float]:
    x_min, y_min, x_max, y_max = bbox
    return [
        (x_min - specimen["x_offset"]) * specimen["x_scale"],
        y_min * specimen["y_scale"],
        (x_max - specimen["x_offset"]) * specimen["x_scale"],
        y_max * specimen["y_scale"],
    ]


def _flybutter_boxes_for_specimen(
    record: dict[str, Any], position: str, specimen: dict[str, Any]
) -> list[dict[str, Any]]:
    boxes = []
    for feature in record["diagnostic_features"]:
        coordinates = feature["coordinates_by_specimen"].get(position, {})
        for wing_slot, slot in coordinates.items():
            if slot.get("state") != "present":
                continue
            components = slot.get("component_bboxes") or [{"bbox": slot["bbox"]}]
            for component in components:
                boxes.append(
                    {
                        "bbox_local_pixels": _flybutter_local_box(
                            component["bbox"], specimen
                        ),
                        "feature": feature.get("feature_id"),
                        "short_label": feature.get("short_label"),
                        "wing_slot": wing_slot,
                        "label": feature.get("description"),
                        "importance": component.get("area_px", slot.get("area_px")),
                        "specimen": position,
                    }
                )
    return boxes


def _iter_flybutter_annotation_outcomes(
    download_dir: Path, *, user_id: str
) -> Iterable[ExplainOutcome]:
    image_id_by_path = _read_image_id_by_path(download_dir)
    with (download_dir / FLYBUTTER_BATCH_FILENAME).open(encoding="utf-8") as handle:
        records_by_triple = json.load(handle)

    for triple_key, record in sorted(records_by_triple.items()):
        triple_id = _triple_id(record.get("image_id", triple_key))
        specimen_info = _flybutter_specimen_info(
            download_dir, triple_id, record, image_id_by_path
        )
        image_ids_by_label = {
            specimen["label"]: specimen["image_id"]
            for specimen in specimen_info.values()
        }
        positions_by_label = {
            specimen["label"]: position
            for position, specimen in specimen_info.items()
        }
        for selected_label in sorted(image_ids_by_label):
            annotated_image = image_ids_by_label[selected_label]
            reference_images = [
                image_ids_by_label[label]
                for label in sorted(image_ids_by_label)
                if label != selected_label
            ]
            position = positions_by_label[selected_label]
            boxes = _flybutter_boxes_for_specimen(
                record, position, specimen_info[position]
            )

            yield ExplainOutcome(
                dataset_id=DatasetId.FLYBUTTER,
                annotated_image=annotated_image,
                reference_image1=reference_images[0],
                reference_image2=reference_images[1],
                user=_ai_choice_user_id(user_id, selected_label),
                explanation=record.get("basis") or "AI bounding-box solution.",
                annotation=_annotation_from_boxes(boxes),
            )


def iter_ai_annotation_outcomes(
    download_dir: Path | None = None, *, user_id: str = AI_USER_ID
) -> Iterable[ExplainOutcome]:
    if download_dir is None:
        for default_download_dir in (
            DATA_DIR / DatasetId.BUTTERFLY / "download",
            DATA_DIR / DatasetId.FLYBUTTER / "download",
        ):
            yield from iter_ai_annotation_outcomes(
                default_download_dir, user_id=user_id
            )
        return

    dataset_id = _dataset_id_from_download_dir(download_dir)
    if dataset_id == DatasetId.FLYBUTTER:
        yield from _iter_flybutter_annotation_outcomes(download_dir, user_id=user_id)
    else:
        yield from _iter_butterfly_annotation_outcomes(
            download_dir, user_id=user_id, dataset_id=dataset_id
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
    outcomes = list(iter_ai_annotation_outcomes(download_dir, user_id=user_id))
    for outcome_user_id in sorted({outcome.user for outcome in outcomes}):
        ensure_ai_user(storage, user_id=outcome_user_id)
    for outcome in outcomes:
        storage.upsert_explain_outcome(outcome)
    return outcomes
