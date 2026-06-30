from __future__ import annotations

from pathlib import Path
from typing import Literal

DatasetId = Literal["butterfly_test2"]

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def main(dataset_id: DatasetId) -> None:
    dataset_dir = DATA_DIR / dataset_id
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        download(dataset_id)
        assert images_dir.exists(), (
            f"Failed to download images for dataset {dataset_id}"
        )


def download(dataset_id: DatasetId) -> None:
    raise NotImplementedError("The download function is not implemented yet.")


if __name__ == "__main__":
    main()
