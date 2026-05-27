from argparse import ArgumentParser, Namespace
from os.path import join

import mlflow
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR
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
    parser.add_argument('--mlflow_address', type=str, default="http://host.docker.internal:8081")
    parser.add_argument('--experiment_name', type=str, default="ImageNet1000")
    parser.add_argument('--resume_from', type=str, default=None)
    return parser.parse_args()


def train_loop(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()

    running_loss, correct1, correct5, total = 0.0, 0.0, 0.0, 0
    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Training epoch {epoch}")
    for i, data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, pred = output.topk(5, dim=1, largest=True, sorted=True)
        labels_reshaped = labels.view(-1, 1)

        correct1 += pred[:, :1].eq(labels_reshaped).sum().item()
        correct5 += pred.eq(labels_reshaped).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss / i:.3f}",
            "top1_acc": f"{correct1 / total:.4f}",
            "top5_acc": f"{correct5 / total:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
        })
    return running_loss / len(dataloader), correct1 / total, correct5 / total


def val_loop(dataloader, model, loss_fn, device, epoch):
    model.eval()

    running_loss, correct1, correct5, total = 0.0, 0.0, 0.0, 0
    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Validation {epoch}")
    with torch.no_grad():
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)

            output = model(inputs)
            running_loss += loss_fn(output, labels).item()

            _, pred = output.topk(5, dim=1, largest=True, sorted=True)
            labels_reshaped = labels.view(-1, 1)

            correct1 += pred[:, :1].eq(labels_reshaped).sum().item()
            correct5 += pred.eq(labels_reshaped).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{running_loss / i:.3f}",
                "top1_acc": f"{correct1 / total:.4f}",
                "top5_acc": f"{correct5 / total:.4f}",
            })
    return running_loss / len(dataloader), correct1 / total, correct5 / total


def save_checkpoint(path: str, model: AlexNet, optimizer: Optimizer, scheduler: LRScheduler, epoch: int,
                    val_top1_acc: float) -> None:
    """
    Save full checkpoint with ability to resume training.

    :param path: Path to save checkpoint.
    :param model: AlexNet model object.
    :param optimizer: Optimizer object.
    :param scheduler: Scheduler object.
    :param epoch: Epoch number.
    :param val_top1_acc: Top-1 accuracy of the validation epoch.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_top1_acc": val_top1_acc,
    }
    torch.save(checkpoint, path)


def main():
    """Parse arguments, prepare data/loaders, and train AlexNet on ImageNet.

    Executes a full training loop: initializes model/optimizer/scheduler,
    iterates over epochs with train/val phases, logs metrics, and saves
    checkpoints after each epoch.
    """
    args = parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_address)
    mlflow_params = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
    }

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

    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(
        optimizer,
        milestones=(int(args.epochs * 0.25), int(args.epochs * 0.5), int(args.epochs * 0.75)),
        gamma=(1 / 250) ** (1 / 3),
    )

    start_epoch = 1
    best_val_top1_acc = float("-inf")
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device)

        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_top1_acc = checkpoint["best_val_top1_acc"]

        print(
            f"Resumed from {args.resume_from}: "
            f"last_epoch={checkpoint['epoch']}, "
            f"start_epoch={start_epoch}, "
            f"best_val_top1_acc={best_val_top1_acc:.4f}"
        )

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=net.__class__.__name__):
        mlflow.log_params(mlflow_params)
        best_model_name = f"alexnet_imagenet1000_best_batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
        last_model_name = f"alexnet_imagenet1000_last_batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_top1_acc, train_top5_acc = train_loop(trainloader, net, criterion, optimizer, device,
                                                                    epoch)
            val_loss, val_top1_acc, val_top5_acc = val_loop(valloader, net, criterion, device, epoch)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_top1_acc": train_top1_acc,
                    "train_top5_acc": train_top5_acc,
                    "val_loss": val_loss,
                    "val_top1_acc": val_top1_acc,
                    "val_top5_acc": val_top5_acc,
                    "lr": current_lr,
                },
                step=epoch,
            )

            last_model_path = join(args.save_model_path, last_model_name)
            is_best = val_top1_acc > best_val_top1_acc
            if is_best:
                best_val_top1_acc = val_top1_acc

            # save last checkpoint every epoch for resume
            save_checkpoint(
                path=last_model_path,
                model=net,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_top1_acc=best_val_top1_acc,
            )
            mlflow.log_artifact(last_model_path)

            if is_best:
                best_model_path = join(args.save_model_path, best_model_name)
                save_checkpoint(
                    path=best_model_path,
                    model=net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_top1_acc=best_val_top1_acc,
                )
                mlflow.log_artifact(best_model_path)

                tqdm.write(f"New best model: epoch={epoch}, val_top1_acc={val_top1_acc:.4f}")
    print(f"Finished Training")


if __name__ == "__main__":
    main()
