from argparse import ArgumentParser, Namespace
from os.path import join
from contextlib import nullcontext

import mlflow
import torch
from torch import nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import random_split, Subset, DataLoader
from torchvision.datasets import GTSRB
from torchvision.models import AlexNet, resnet18, resnet34, resnet50, resnet101, resnet152
from tqdm import tqdm

from src.transforms import get_train_transforms, get_val_transforms


def parse_args() -> Namespace:
    """Parse command-line arguments for the training script.

    :return: Parsed arguments namespace containing training hyperparameters.
    """
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="artifacts/datasets/GTSRB")
    parser.add_argument("--model", type=str, help="Model architecture to use.", required=True)
    parser.add_argument("--save_model_path", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--resume_from", type=str, default=None)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # Scheduler
    parser.add_argument("--mode", type=str, default="min")
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler_threshold", type=float, default=0.001)
    parser.add_argument("--threshold_mode", type=str, default="abs")
    parser.add_argument("--min_lr", type=float, default=1e-5)

    # MlFlow
    parser.add_argument("--mlflow_address", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="GTSRB")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def train_loop(dataloader, model, loss_fn, optimizer, device, epoch):
    """Run one training epoch.

    Performs forward pass, loss computation, backpropagation, optimizer step,
    and calculates average loss, Top-1 accuracy, and Top-5 accuracy.

    :param dataloader: DataLoader with training batches.
    :param model: Model to train.
    :param loss_fn: Loss function.
    :param optimizer: Optimizer used to update model parameters.
    :param device: Device on which tensors and model are located.
    :param epoch: Current epoch number.
    :return: Tuple containing average loss, Top-1 accuracy, and Top-5 accuracy.
    """
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
            "lr": f"{optimizer.param_groups[0]["lr"]:.5f}",
        })
    return running_loss / len(dataloader), correct1 / total, correct5 / total


def val_loop(dataloader, model, loss_fn, device, epoch):
    """Run one validation epoch.

    Performs inference without gradient calculation and computes average loss,
    Top-1 accuracy, and Top-5 accuracy on the validation dataset.

    :param dataloader: DataLoader with validation batches.
    :param model: Model to evaluate.
    :param loss_fn: Loss function.
    :param device: Device on which tensors and model are located.
    :param epoch: Current epoch number.
    :return: Tuple containing average loss, Top-1 accuracy, and Top-5 accuracy.
    """
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


def save_checkpoint(path: str, model: Module, optimizer: Optimizer, scheduler: LRScheduler, epoch: int,
                    best_val_top1_acc: float) -> None:
    """Save full checkpoint with ability to resume training.

    :param path: Path to save checkpoint.
    :param model: Model object.
    :param optimizer: Optimizer object.
    :param scheduler: Scheduler object.
    :param epoch: Epoch number.
    :param best_val_top1_acc: Best validation Top-1 accuracy so far.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_top1_acc": best_val_top1_acc,
    }
    torch.save(checkpoint, path)


def main():
    """Parse arguments, prepare data/loaders, and train model on GTSRB.

    Executes a full training loop: initializes model/optimizer/scheduler,
    iterates over epochs with train/val phases, logs metrics, and saves
    checkpoints after each epoch.
    """
    args = parse_args()
    use_mlflow = args.mlflow_address is not None

    # MLflow setup
    mlflow_params = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
    }
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_address)
        mlflow.set_experiment(args.experiment_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Device: {device}")

    dataset = GTSRB(root=args.data_root, split="train", transform=get_train_transforms(), download=True)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_subset, val_subset = random_split(dataset, (n_train, n_val))

    train_dataset = GTSRB(root=args.data_root, split="train", transform=get_train_transforms())
    val_dataset = GTSRB(root=args.data_root, split="train", transform=get_val_transforms())

    trainset = Subset(train_dataset, train_subset.indices)
    valset = Subset(val_dataset, val_subset.indices)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    tqdm.write(f"Train dataset size: {len(trainset)}")

    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    tqdm.write(f"Val dataset size: {len(valset)}")

    # Remove 'weights' param if you want to train from scratch
    net = models_dict[args.model](weights="IMAGENET1K_V1")

    ## Uncomment for freezing backbone
    # for param in net.parameters():
    #     param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, len(set([label for _, label in dataset])))
    net.to(device)

    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=args.mode,
        factor=args.factor,
        patience=args.patience,
        threshold=args.scheduler_threshold,
        threshold_mode=args.threshold_mode,
        min_lr=args.min_lr,
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

        tqdm.write(
            f"Resumed from {args.resume_from}: "
            f"last_epoch={checkpoint["epoch"]}, "
            f"start_epoch={start_epoch}, "
            f"best_val_top1_acc={best_val_top1_acc:.4f}"
        )

    run_context = mlflow.start_run(run_name=args.model) if use_mlflow else nullcontext()
    with run_context:
        if use_mlflow:
            mlflow.log_params(mlflow_params)
        mlflow.log_params(mlflow_params)
        best_model_name = f"{args.model.lower()}_gtsrb_best_batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
        last_model_name = f"{args.model.lower()}_gtsrb_last_batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_top1_acc, train_top5_acc = train_loop(trainloader, net, criterion, optimizer, device,
                                                                    epoch)
            val_loss, val_top1_acc, val_top5_acc = val_loop(valloader, net, criterion, device, epoch)

            val_top1_error = 1.0 - val_top1_acc
            scheduler.step(val_top1_error)
            current_lr = optimizer.param_groups[0]["lr"]

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
                best_val_top1_acc=best_val_top1_acc,
            )

            if use_mlflow:
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
                mlflow.log_artifact(last_model_path)

            if is_best:
                best_model_path = join(args.save_model_path, best_model_name)
                save_checkpoint(
                    path=best_model_path,
                    model=net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_top1_acc=best_val_top1_acc,
                )
                if use_mlflow:
                    mlflow.log_artifact(best_model_path)

                tqdm.write(f"New best model: epoch={epoch}, val_top1_acc={val_top1_acc:.4f}")
    tqdm.write(f"Finished Training")


if __name__ == "__main__":
    models_dict = {
        "AlexNet": AlexNet,
        "ResNet18": resnet18,
        "ResNet34": resnet34,
        "ResNet50": resnet50,
        "ResNet101": resnet101,
        "ResNet152": resnet152,
    }
    main()
