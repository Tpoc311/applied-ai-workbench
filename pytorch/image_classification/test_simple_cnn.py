import argparse

import torch

from src.datasets.cifar10 import create_test_dataloader
from src.models.simple_net import Net


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    :return: Parsed arguments namespace containing data paths and batch settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="artifacts/datasets/CIFAR10")
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """
    Evaluate the trained model on the CIFAR-10 test dataset.

    Loads model weights, computes predictions without gradient tracking,
    and prints the final top-1 accuracy.
    """
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    testloader, images_count = create_test_dataloader(args.data_root, args.batch_size, args.num_workers)

    net = Net()
    net.load_state_dict(torch.load(args.load_model_path, weights_only=True))
    net.to(device)
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {images_count} test images: {100 * correct // total} %')


if __name__ == '__main__':
    main()
