from __future__ import annotations

from pathlib import Path

import streamlit as st
from typing_extensions import assert_never

from diff_detect.models import (
    Dataset,
    DatasetId,
    ExplainChallengeId,
    ExplainDifferenceChallenge,
    ExplainDiffernceTask,
    Image,
    ImageId,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


# def get_challenge(
#     challenge_id: ChallengeId, data_dir: Path = DATA_DIR
# ) -> tuple[dict[DatasetId, Dataset], SelectionChallenge]:
#     if challenge_id in ("select_dummy", "select_butterfly_easy", "select_butterfly_difficult"):
#         return _get_selection_challenge(challenge_id, data_dir)
#     elif challenge_id in ("rate_dummy", "rate_butterfly_easy", "rate_butterfly_difficult"):
#         return _get_rating_challenge(challenge_id, data_dir)
#     else:
#         assert_never(challenge_id)
@st.cache_data
def get_explain_challenge(
    challenge_id: ExplainChallengeId, data_dir: Path
) -> tuple[dict[DatasetId, Dataset], ExplainDifferenceChallenge]:
    dataset_id = "butterfly"
    if challenge_id == "select_dummy":
        index_path = Path("index_dummy.csv")
        difficulty = 0.5
    elif challenge_id == "select_butterfly_easy":
        index_path = Path("index_easy.csv")
        difficulty = 0.25
    elif challenge_id == "select_butterfly_difficult":
        index_path = Path(
            "index_difficult.csv",
        )
        difficulty = 0.75
    else:
        assert_never(challenge_id)

    dataset_dir = data_dir / dataset_id
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
            image = Image(
                id=ImageId(f"{dataset_id}/{image_id}"),
                source=(download_dir / path).relative_to(dataset_dir).as_posix(),
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
            images[image.id] = image
            image_groups.setdefault(task_id, []).append(image)

    assert all(len(images) == 3 for images in image_groups.values()), (
        "Each task must have exactly 3 images."
    )

    tasks: list[ExplainDiffernceTask] = []
    for task_id in sorted(image_groups.keys()):
        group = image_groups[task_id]
        tasks.extend(
            [
                ExplainDiffernceTask(
                    annotated_image=group[0].id,
                    reference_image1=group[1].id,
                    reference_image2=group[2].id,
                ),
                ExplainDiffernceTask(
                    annotated_image=group[1].id,
                    reference_image1=group[2].id,
                    reference_image2=group[0].id,
                ),
                ExplainDiffernceTask(
                    annotated_image=group[2].id,
                    reference_image1=group[0].id,
                    reference_image2=group[1].id,
                ),
            ]
        )

    return {dataset_id: Dataset(images=images)}, ExplainDifferenceChallenge(
        challenge_id=challenge_id,
        tasks=tasks,
    )


def download(dataset_id: DatasetId, path: Path) -> None:
    assert not path.exists(), f"Download directory {path} already exists."
    raise NotImplementedError("The download function is not implemented yet.")

    assert path.exists(), f"Failed to download dataset {dataset_id} to {path}."
