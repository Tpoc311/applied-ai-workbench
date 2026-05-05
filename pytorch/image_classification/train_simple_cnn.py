import argparse
from os.path import join
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.cifar10 import create_train_dataloader
from src.models.simple_net import Net


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    :return: Parsed arguments namespace containing training hyperparameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="artifacts/datasets/CIFAR10")
    parser.add_argument('--save_model_path', type=str, default=".")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    """Train the CNN on CIFAR-10 and save the final model checkpoint.

    Iterates over the training set for a specified number of epochs,
    computes loss, updates weights via SGD, and prints epoch statistics.
    """
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trainloader, _ = create_train_dataloader(args.data_root, args.batch_size, args.num_workers)

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    start_time = time()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, start=1):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'{epoch}, loss: {running_loss / i:.3f}')
    end_time = time()
    print('Finished Training. Total time elapsed: {0:.3f} minutes'.format((end_time - start_time) / 60))

    model_name = f"simple-net_cifar10_epoch{epoch}_batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
    torch.save(net.state_dict(), join(args.save_model_path, model_name))


if __name__ == '__main__':
    main()
