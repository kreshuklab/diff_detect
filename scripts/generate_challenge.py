from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from typing_extensions import assert_never

from diff_detect.models import (
    ChallengeId,
    Dataset,
    Image,
    SelectionChallenge,
    SelectionTask,
)

DatasetId = Literal["butterfly"]

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def main(challenge_id: ChallengeId) -> None:
    dataset_id = "butterfly"
    if challenge_id == "dummy":
        index_path = "index_dummy.csv"
        difficulty = 0.5
    elif challenge_id == "butterfly_easy":
        index_path = "index_easy.csv"
        difficulty = 0.25
    elif challenge_id == "butterfly_difficult":
        difficulty = 0.75
        index_path = "index_difficult.csv"
    else:
        assert_never(challenge_id)

    dataset_dir = DATA_DIR / dataset_id
    download_dir = dataset_dir / "download"
    if not download_dir.exists():
        download(dataset_id, path=download_dir)

    images: list[Image] = []
    image_groups: dict[str, list[Image]] = {}

    header = None
    delimiter = ";"
    with (download_dir / index_path).open() as f:
        for line in f:
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
                dataset_id=dataset_id,
                image_id=image_id,
                path=(download_dir / path).relative_to(dataset_dir),
                hash_kwargs=None,
                image_info={
                    "mimic_group": mimic_group,
                    "species": species,
                    "subspecies": subspecies,
                    "sex": sex,
                    "hybrid_stat": hybrid_stat,
                    "file_url": file_url,
                },
                image_group=(task_id,),
            )
            images.append(image)
            image_groups.setdefault(task_id, []).append(image)

        tasks: list[SelectionTask] = []
        for task_id in sorted(image_groups.keys()):
            tasks.append(
                SelectionTask(
                    images=tuple(image_groups[task_id]),
                    difficulty=difficulty,
                )
            )

        challenge = SelectionChallenge(
            dataset_id=dataset_id, challenge_id=challenge_id, tasks=tasks
        )
        challenge_path = dataset_dir / f"{challenge_id}.json"
        with challenge_path.open("wb") as f:
            f.write(
                json.dumps(challenge.model_dump(mode="json"), indent=2).encode("utf-8")
            )

        dataset = Dataset(images)
        dataset_path = dataset_dir / f"{dataset_id}.json"
        with dataset_path.open("wb") as f:
            f.write(
                json.dumps(dataset.model_dump(mode="json"), indent=2).encode("utf-8")
            )


def download(dataset_id: DatasetId, path: Path) -> None:
    assert not path.exists(), f"Download directory {path} already exists."
    raise NotImplementedError("The download function is not implemented yet.")

    assert path.exists(), f"Failed to download dataset {dataset_id} to {path}."


if __name__ == "__main__":
    main("dummy")
    main("butterfly_easy")
    main("butterfly_difficult")
