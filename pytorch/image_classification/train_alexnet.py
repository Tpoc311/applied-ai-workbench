import argparse
from os.path import join
from time import time

import torch
import torchvision
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import AlexNet
from torchvision.transforms.v2 import Resize, RandomCrop, RandomHorizontalFlip, ToImage, ToDtype, Normalize, Compose


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    :return: Parsed arguments namespace containing training hyperparameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="artifacts/datasets/ImageNet/ILSVRC2012")
    parser.add_argument('--save_model_path', type=str, default=".")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    return parser.parse_args()


def decode_image(path: str) -> torch.Tensor:
    image = torchvision.io.decode_image(path)
    if image.shape[0] == 4:
        image = image[:3, :, :]
    elif image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


def get_transforms() -> Compose:
    """
    Return preprocessing transforms for AlexNet.

    :return: Composed transformation sequence ready to be applied to dataset.
    """
    return Compose([Resize(256),
                    RandomCrop((224, 224)),
                    RandomHorizontalFlip(),
                    ToImage(),
                    ToDtype(torch.float32, scale=True),
                    Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ])


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = ImageNet(root=args.data_root, split='val', loader=decode_image, transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('Train dataset size:', len(train_dataset))

    net = AlexNet()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    total_start_time = time()
    for epoch in range(1, args.epochs + 1):
        start_time = time()
        running_loss = 0.0
        i = 1
        for data in tqdm(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            i += 1
        print(f"{epoch}, loss: {running_loss / i:.3f}, took minutes: {round((time() - start_time) / 60, ndigits=2)}")
    print(f"Finished Training. Total time elapsed: {round((time() - total_start_time) / 60, ndigits=2)} minutes")

    model_name = f"alexnet_imagenet1000_epoch{epoch}_batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
    torch.save(net.state_dict(), join(args.save_model_path, model_name))


if __name__ == "__main__":
    main()
