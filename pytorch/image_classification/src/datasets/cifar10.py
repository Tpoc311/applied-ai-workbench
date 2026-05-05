from torch import float32

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import ToImage, ToDtype, Normalize, Compose


def get_transforms() -> Compose:
    """Return simple preprocessing transforms for CIFAR-10.

    The pipeline converts input images to tensors and normalizes pixel values
    to the [-1, 1] range.

    :return: Composed transformation sequence ready to be applied to dataset.
    """
    return Compose([ToImage(),
                    ToDtype(float32, scale=True),
                    Normalize(mean=(0.49139968, 0.48215827, 0.44653124),
                              std=(0.24703224, 0.24348514, 0.26158786))])


def create_train_dataloader(data_root: str, batch_size: int, num_workers: int) -> tuple[DataLoader, int]:
    """Create a DataLoader for the CIFAR-10 train split.

    :param data_root: Root directory where the CIFAR-10 dataset is stored.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of subprocesses used for parallel data loading.
    :return: Tuple containing the training DataLoader and the total dataset size.
    """
    trainset = CIFAR10(root=data_root, train=True, download=False, transform=get_transforms())
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers), len(trainset)


def create_test_dataloader(data_root: str, batch_size: int, num_workers: int) -> tuple[DataLoader, int]:
    """Create a DataLoader for the CIFAR-10 test split.

    :param data_root: Root directory where the CIFAR-10 dataset is stored.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of subprocesses used for parallel data loading.
    :return: Tuple containing the test DataLoader and the total dataset size.
    """
    testset = CIFAR10(root=data_root, train=False, download=False, transform=get_transforms())
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers), len(testset)
