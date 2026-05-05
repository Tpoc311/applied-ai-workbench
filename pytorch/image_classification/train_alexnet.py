from argparse import ArgumentParser, Namespace
from os.path import join

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import AlexNet
from tqdm import tqdm

from src.transforms.alexnet import get_train_transforms, get_val_transforms
from src.utils import decode_image


def parse_args() -> Namespace:
    """Parse command-line arguments for the training script.

    :return: Parsed arguments namespace containing training hyperparameters.
    """
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default="artifacts/datasets/ImageNet/ILSVRC2012")
    parser.add_argument('--save_model_path', type=str, default=".")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    return parser.parse_args()


def train_loop(dataloader, model, loss_fn, optimizer, device, epoch):
    """Run one training epoch for the given model.

    Performs forward/backward pass, updates weights, and tracks loss/accuracy.

    :param dataloader: DataLoader yielding training batches (images, labels).
    :param model: Neural network to train (will be set to .train() mode).
    :param loss_fn: Loss function for computing training loss.
    :param optimizer: Optimizer for updating model parameters.
    :param device: PyTorch device (cpu/cuda) for tensor placement.
    :param epoch: Current epoch number (for progress bar description).
    :return: Tuple of (average_loss, accuracy) over the training dataloader.
    """
    model.train()

    running_loss, correct, total = 0.0, 0.0, 0
    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Training epoch {epoch}")
    for i, data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (output.argmax(1) == labels).type(torch.float).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss / i:.3f}",
            "acc": f"{correct / total:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
        })
    return running_loss / len(dataloader), correct / total


def val_loop(dataloader, model, loss_fn, device, epoch):
    """Run one validation epoch for the given model.

    Evaluates model performance without gradient computation.

    :param dataloader: DataLoader yielding validation batches (images, labels).
    :param model: Neural network to evaluate (will be set to .eval() mode).
    :param loss_fn: Loss function for computing validation loss.
    :param device: PyTorch device (cpu/cuda) for tensor placement.
    :param epoch: Current epoch number (for progress bar description).
    :return: Tuple of (average_loss, accuracy) over the validation dataloader.
    """
    model.eval()

    running_loss, correct, total = 0.0, 0.0, 0
    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Validation {epoch}")
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
    """Parse arguments, prepare data/loaders, and train AlexNet on ImageNet.

    Executes a full training loop: initializes model/optimizer/scheduler,
    iterates over epochs with train/val phases, logs metrics, and saves
    checkpoints after each epoch.
    """
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trainset = ImageNet(root=args.data_root, split='train', loader=decode_image, transform=get_train_transforms())
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('Train dataset size:', len(trainset))

    valset = ImageNet(root=args.data_root, split='val', loader=decode_image, transform=get_val_transforms())
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Val dataset size:', len(valset))

    net = AlexNet()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(
        optimizer,
        milestones=(int(args.epochs * 0.25), int(args.epochs * 0.5), int(args.epochs * 0.75)),
        gamma=(1 / 250) ** (1 / 3),
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_loop(trainloader, net, criterion, optimizer, device, epoch)
        val_loss, val_acc = val_loop(valloader, net, criterion, device, epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        tqdm.write(
            f"epoch={epoch:03d}, "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"lr={current_lr:.6f}"
        )

        model_name = f"alexnet_imagenet1000_epoch{epoch}_batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
        torch.save(net.state_dict(), join(args.save_model_path, model_name))
    print(f"Finished Training")


if __name__ == "__main__":
    main()
