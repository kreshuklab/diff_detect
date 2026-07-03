from .models import (
    Dataset,
    DatasetId,
    Image,
    ImageId,
)


def get_image(
    datasets: dict[DatasetId, Dataset],
    dataset_id: DatasetId,
    image_id: ImageId,
) -> Image:
    if dataset_id not in datasets:
        raise KeyError(f"Dataset '{dataset_id}' missing in active task.")

    dataset = datasets[dataset_id]
    image = dataset.images.get(image_id)
    if image is None:
        raise KeyError(
            f"Image '{image_id}' was not found in dataset '{dataset_id}'."
            + f"available images: {list(dataset.images.keys())}"
        )

    return image
