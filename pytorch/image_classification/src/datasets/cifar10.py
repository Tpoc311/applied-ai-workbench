from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms.transforms import ToTensor, Normalize


def get_transforms() -> transforms.Compose:
    """
    Return simple preprocessing transforms for CIFAR-10.

    The pipeline converts input images to tensors and normalizes pixel values
    to the [-1, 1] range.

    :return: Composed transformation sequence ready to be applied to dataset.
    """
    return transforms.Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def create_train_dataloader(data_root: str, batch_size: int, num_workers: int) -> tuple[DataLoader, int]:
    """
    Create a DataLoader for the CIFAR-10 train split.

    :param data_root: Root directory where the CIFAR-10 dataset is stored.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of subprocesses used for parallel data loading.
    :return: Tuple containing the training DataLoader and the total dataset size.
    """
    trainset = CIFAR10(root=data_root, train=True, download=False, transform=get_transforms())
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers), len(trainset)


def create_test_dataloader(data_root: str, batch_size: int, num_workers: int) -> tuple[DataLoader, int]:
    """
    Create a DataLoader for the CIFAR-10 test split.

    :param data_root: Root directory where the CIFAR-10 dataset is stored.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of subprocesses used for parallel data loading.
    :return: Tuple containing the test DataLoader and the total dataset size.
    """
    testset = CIFAR10(root=data_root, train=False, download=False, transform=get_transforms())
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers), len(testset)
