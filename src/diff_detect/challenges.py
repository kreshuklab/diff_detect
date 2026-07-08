from __future__ import annotations

import csv
from pathlib import Path

import streamlit as st
from PIL import Image as PILImage
from typing_extensions import assert_never

from diff_detect.models import (
    Dataset,
    DatasetId,
    ExplainChallenge,
    ExplainChallengeId,
    ExplainTask,
    Image,
    ImageId,
    UserRole,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def _resolve_download_source(dataset_dir: Path, download_dir: Path, path: str) -> str:
    candidate = download_dir / path
    if not candidate.exists():
        path_obj = Path(path)
        candidate = download_dir / path_obj.parent.name / path_obj.name

    assert candidate.exists(), f"Image file not found: {candidate}"
    return candidate.relative_to(dataset_dir.parent).as_posix()


def _image_from_index_row(
    dataset_id: DatasetId, dataset_dir: Path, download_dir: Path, row: dict[str, str]
) -> tuple[str, Image]:
    task_id = row["task_id"]
    path = row["path"]
    image = Image(
        dataset_id=dataset_id,
        image_id=ImageId(Path(path).stem),
        source=_resolve_download_source(dataset_dir, download_dir, path),
        image_info={
            "mimic_group": row["mimic_group"],
            "species": row["species"],
            "subspecies": row["subspecies"],
            "sex": row["Sex"],
            "hybrid_stat": row["hybrid_stat"],
            "file_url": row["file_url"],
            "camid": row["CAMID"],
        },
        image_group=task_id,
    )
    return task_id, image


def get_available_explain_challenges(
    user_role: UserRole,
) -> tuple[dict[DatasetId, Dataset], dict[ExplainChallengeId, ExplainChallenge]]:
    challenge_ids: list[ExplainChallengeId] = [
        "explain_butterfly_easy",
        "explain_butterfly_difficult",
    ]
    if user_role == UserRole.MAINTAINER:
        challenge_ids.insert(0, "explain_dummy")

    challenges: dict[ExplainChallengeId, ExplainChallenge] = {}
    merged_datasets: dict[DatasetId, Dataset] = {}
    for c_id in challenge_ids:
        datasets, c = get_explain_challenge(c_id)
        challenges[c.id] = c
        for d_id, d in datasets.items():
            if d_id in merged_datasets:
                merged_datasets[d_id].images.update(d.images)
            else:
                merged_datasets[d_id] = d

    return merged_datasets, challenges


@st.cache_data
def get_explain_challenge(
    challenge_id: ExplainChallengeId,
) -> tuple[dict[DatasetId, Dataset], ExplainChallenge]:
    dataset_id = DatasetId.BUTTERFLY
    if challenge_id == "explain_dummy":
        index_path = Path("index_dummy.csv")
    elif challenge_id == "explain_butterfly_easy":
        index_path = Path("easy.csv")
    elif challenge_id == "explain_butterfly_difficult":
        index_path = Path("difficult.csv")
    else:
        assert_never(challenge_id)

    dataset_dir = DATA_DIR / dataset_id
    download_dir = dataset_dir / "download"
    if not download_dir.exists():
        download(dataset_id, path=download_dir)

    images: dict[ImageId, Image] = {}
    image_groups: dict[str, list[Image]] = {}

    if not (download_dir / index_path).exists() and not index_path.stem.startswith(
        "index_"
    ):
        index_path = Path(f"index_{index_path.stem}.csv")

    with (download_dir / index_path).open(encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        delimiter = ";" if ";" in sample and "," not in sample else ","
        for row in csv.DictReader(handle, delimiter=delimiter):
            task_id, image = _image_from_index_row(
                dataset_id, dataset_dir, download_dir, row
            )
            images[image.image_id] = image
            image_groups.setdefault(task_id, []).append(image)

    assert all(len(images) == 3 for images in image_groups.values()), (
        "Each task must have exactly 3 images."
    )

    tasks: list[ExplainTask] = []
    for task_id in sorted(image_groups.keys()):
        group = image_groups[task_id]
        tasks.append(
            ExplainTask(
                dataset_id=dataset_id,
                annotated_image=group[0].image_id,
                reference_image1=group[1].image_id,
                reference_image2=group[2].image_id,
            )
        )

    return {dataset_id: Dataset(images=images)}, ExplainChallenge(
        id=challenge_id,
        tasks=tasks,
    )


def download(dataset_id: DatasetId, path: Path) -> None:
    assert not path.exists(), f"Download directory {path} already exists."
    raise NotImplementedError("The download function is not implemented yet.")

    assert path.exists(), f"Failed to download dataset {dataset_id} to {path}."


# @st.cache_data(max_entries=100) # cache here to dynamically adapt background to light/dark theme choice
def _load_study_image_impl(image: Image) -> PILImage.Image:
    path = DATA_DIR / image.source
    assert path.exists(), f"Image file not found: {path}"
    rgba_image = PILImage.open(path).convert("RGBA")
    return rgba_image


@st.cache_data(max_entries=100)
def load_study_image(image: Image) -> PILImage.Image:
    rgba_image = _load_study_image_impl(image)
    theme_base = st.get_option("theme.base")
    if theme_base == "light":
        background = PILImage.new("RGBA", rgba_image.size, (245, 247, 250, 255))
    else:
        background = PILImage.new("RGBA", rgba_image.size, (0, 0, 0, 255))
    return PILImage.alpha_composite(background, rgba_image).convert("RGB")
