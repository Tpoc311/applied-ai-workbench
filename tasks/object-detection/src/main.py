import os

import torch
import torchvision

from dataset.dataset import KITTIDataset


def main():
    print(f"pytorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"User: {os.getlogin()}")

    train_root = "artifacts/datasets/KITTI/object/2d-object/data_object_image_2/training"
    train_dataset = KITTIDataset(root=train_root)
    print(f"Train size: {len(train_dataset)}")

    image, label = train_dataset[0]
    print(f"Image sample: {image}, Label sample: {label}")


if __name__ == "__main__":
    main()
