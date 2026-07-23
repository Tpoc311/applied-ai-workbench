from typing import Any

from torch import Tensor


def collate_fn(batch: list[tuple[Tensor, dict[str, Any]]]) -> tuple[list[Tensor], list[dict[str, Any]]]:
    """Collate detection samples into lists of images and targets.

    Faster R-CNN expects a list of image tensors and a list of target
    dictionaries instead of one stacked tensor batch.

    :param batch: Samples returned by the dataset.
    :return: Images and their corresponding target dictionaries.
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets
