from torch import float32
from torchvision.transforms.v2 import Resize, RandomCrop, RandomHorizontalFlip, ToDtype, Normalize, Compose, CenterCrop


def get_train_transforms() -> Compose:
    """Create a Compose of transforms for the training stage.

    :return: A Compose object with resizing, random cropping, dtype conversion, and normalization ready for validation.
    """
    return Compose([
        Resize(256),
        RandomCrop((224, 224)),
        RandomHorizontalFlip(),
        ToDtype(float32, scale=True),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def get_val_transforms() -> Compose:
    """Create a Compose of transforms for the validation stage.

    :return: A Compose object with resizing, cropping, dtype conversion, and normalization ready for validation.
    """
    return Compose([
        Resize(256),
        CenterCrop((224, 224)),
        ToDtype(float32, scale=True),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
