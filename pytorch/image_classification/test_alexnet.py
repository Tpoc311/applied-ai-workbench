import os
from argparse import ArgumentParser, Namespace
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import AlexNet
from tqdm import tqdm

from src.transforms.alexnet import get_val_transforms
from src.utils import decode_image


def parse_args() -> Namespace:
    """Parse command-line arguments for the testing script.

    :return: Namespace containing parsed CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default="artifacts/datasets/ImageNet/ILSVRC2012")
    parser.add_argument('--models_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def val_loop(dataloader, model, loss_fn, device):
    """Run the validation loop for the AlexNet model.

    :param dataloader: DataLoader yielding validation batches (images, labels).
    :param model: Pre-trained neural network to evaluate.
    :param loss_fn: Loss function used to compute validation loss.
    :param device: PyTorch device (cpu/cuda) for tensor placement.
    :return: Tuple of (average_loss, accuracy) over the entire validation set.
    """
    model.eval()

    running_loss, correct, total = 0.0, 0.0, 0

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Testing")
    with torch.no_grad():
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)

            output = model(inputs)
            running_loss += loss_fn(output, labels).item()
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{running_loss / i:.3f}",
                "acc": f"{correct / total:.4f}",
            })

    return running_loss / len(dataloader), correct / total


def main():
    """Parse arguments, load the validation dataset, and evaluate all saved models.

     Iterates over model checkpoints in the specified directory, loads each AlexNet,
     computes validation loss and accuracy on ImageNet, and writes results to a file.
     """
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    valset = ImageNet(root=args.data_root, split='val', loader=decode_image, transform=get_val_transforms())
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Val dataset size:', len(valset))

    with open("ILSVRC2012_val_images.txt", "w") as f:
        for model_name in sorted(os.listdir(args.models_dir)):
            model_path = join(args.models_dir, model_name)

            net = AlexNet()
            state_dict = torch.load(model_path, map_location=device)
            net.load_state_dict(state_dict)
            net.to(device)

            criterion = nn.CrossEntropyLoss()
            val_loss, val_acc = val_loop(valloader, net, criterion, device)

            f.write(f"model: {model_name}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}.\n")
            tqdm.write(f"model: {model_name}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}.\n")
    tqdm.write(f"Finished Testing")


if __name__ == "__main__":
    main()
