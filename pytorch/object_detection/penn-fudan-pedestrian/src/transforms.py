from torch import float32
from torchvision.transforms.v2 import RandomHorizontalFlip, ToDtype, ToPureTensor, Compose


def get_transform(train):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToDtype(float32, scale=True))
    transforms.append(ToPureTensor())
    return Compose(transforms)
