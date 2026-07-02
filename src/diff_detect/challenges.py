from __future__ import annotations

from pathlib import Path

import streamlit as st
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
        difficulty = 0.5
    elif challenge_id == "explain_butterfly_easy":
        index_path = Path("index_easy.csv")
        difficulty = 0.25
    elif challenge_id == "explain_butterfly_difficult":
        index_path = Path(
            "index_difficult.csv",
        )
        difficulty = 0.75
    else:
        assert_never(challenge_id)

    dataset_dir = DATA_DIR / dataset_id
    download_dir = dataset_dir / "download"
    if not download_dir.exists():
        download(dataset_id, path=download_dir)

    images: dict[ImageId, Image] = {}
    image_groups: dict[str, list[Image]] = {}

    header = None
    delimiter = ";"
    with (download_dir / index_path).open(encoding="utf-8") as handle:
        for line in handle:
            if header is None:
                header = line.strip().split(delimiter)
                assert header == [
                    "task_id",
                    "mimic_group",
                    "species",
                    "subspecies",
                    "path",
                    "CAMID",
                    "Sex",
                    "file_url",
                    "hybrid_stat",
                ]
                continue

            (
                task_id,
                mimic_group,
                species,
                subspecies,
                path,
                image_id,
                sex,
                file_url,
                hybrid_stat,
            ) = line.strip().split(delimiter)
            print("loaded", image_id)
            image = Image(
                dataset_id=dataset_id,
                image_id=ImageId(image_id),
                source=(download_dir / path).relative_to(dataset_dir.parent).as_posix(),
                image_info={
                    "mimic_group": mimic_group,
                    "species": species,
                    "subspecies": subspecies,
                    "sex": sex,
                    "hybrid_stat": hybrid_stat,
                    "file_url": file_url,
                },
                image_group=task_id,
            )
            images[image.image_id] = image
            image_groups.setdefault(task_id, []).append(image)

    assert all(len(images) == 3 for images in image_groups.values()), (
        "Each task must have exactly 3 images."
    )

    tasks: list[ExplainTask] = []
    for task_id in sorted(image_groups.keys()):
        group = image_groups[task_id]
        tasks.extend(
            [
                ExplainTask(
                    dataset_id=dataset_id,
                    annotated_image=group[0].image_id,
                    reference_image1=group[1].image_id,
                    reference_image2=group[2].image_id,
                ),
                ExplainTask(
                    dataset_id=dataset_id,
                    annotated_image=group[1].image_id,
                    reference_image1=group[2].image_id,
                    reference_image2=group[0].image_id,
                ),
                ExplainTask(
                    dataset_id=dataset_id,
                    annotated_image=group[2].image_id,
                    reference_image1=group[0].image_id,
                    reference_image2=group[1].image_id,
                ),
            ]
        )

    return {dataset_id: Dataset(images=images)}, ExplainChallenge(
        id=challenge_id,
        tasks=tasks,
    )


def download(dataset_id: DatasetId, path: Path) -> None:
    assert not path.exists(), f"Download directory {path} already exists."
    raise NotImplementedError("The download function is not implemented yet.")

    assert path.exists(), f"Failed to download dataset {dataset_id} to {path}."
