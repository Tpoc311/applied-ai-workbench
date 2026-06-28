from argparse import ArgumentParser, Namespace
from os.path import join

import mlflow
import torch
from torch import Generator
from torch.nn import Module
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.dataset.penn_fudan import PennFudanDataset
from src.transforms import get_transform
from src.utils import collate_fn


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="artifacts/datasets/PennFudanPed")
    parser.add_argument("--save_model_path", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--resume_from", type=str, default=None)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # Scheduler
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.1)

    # MlFlow
    parser.add_argument("--mlflow_address", type=str, default="http://host.docker.internal:8081")
    parser.add_argument("--experiment_name", type=str, default="PennFudanPed")
    parser.add_argument("--run_name", type=str, default='fasterrcnn_resnet50_fpn')
    return parser.parse_args()


def train_loop(dataloader, model, optimizer, device, epoch):
    model.train()

    loss_sums = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "total_loss": 0.0,
    }

    pbar = tqdm(
        enumerate(dataloader, start=1),
        total=len(dataloader),
        desc=f"Training epoch {epoch}",
    )

    for i, data in pbar:
        images = [image.to(device) for image in data[0]]

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in
                   data[1]]

        optimizer.zero_grad()

        loss_dict = model(images, targets)

        total_loss = sum(loss for loss in loss_dict.values())

        total_loss.backward()
        optimizer.step()

        for name, value in loss_dict.items():
            loss_sums[name] += value.item()
        loss_sums["total_loss"] += total_loss.item()

        pbar.set_postfix({
            "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
            "loss_cls": f"{loss_sums['loss_classifier'] / i:.3f}",
            "loss_box": f"{loss_sums['loss_box_reg'] / i:.3f}",
            "loss_obj": f"{loss_sums['loss_objectness'] / i:.3f}",
            "loss_rpn_box": f"{loss_sums['loss_rpn_box_reg'] / i:.3f}",
            "total_loss": f"{loss_sums['total_loss'] / i:.3f}",
        })

    losses = {
        name: value / len(dataloader)
        for name, value in loss_sums.items()
    }

    return losses


def val_loss_loop(dataloader, model, device, epoch):
    # TODO Make metrics calculation later
    was_training = model.training
    model.train()
    loss_sums = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "total_loss": 0.0,
    }

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Validation epoch {epoch}")

    with torch.no_grad():
        for i, data in pbar:
            images = [image.to(device) for image in data[0]]

            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in target.items()
                }
                for target in data[1]
            ]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            for name, loss_value in loss_dict.items():
                loss_sums[name] += loss_value.item()
            loss_sums["total_loss"] += total_loss.item()

            pbar.set_postfix({
                "loss_cls": f"{loss_sums['loss_classifier'] / i:.3f}",
                "loss_box": f"{loss_sums['loss_box_reg'] / i:.3f}",
                "loss_obj": f"{loss_sums['loss_objectness'] / i:.3f}",
                "loss_rpn_box": f"{loss_sums['loss_rpn_box_reg'] / i:.3f}",
                "total": f"{loss_sums['total_loss'] / i:.3f}",
            })

    model.train(was_training)

    return {name: value / len(dataloader) for name, value in loss_sums.items()}


def save_checkpoint(path: str, model: Module, optimizer: Optimizer, scheduler: StepLR, epoch: int,
                    best_val_loss: float) -> None:
    """
    Save full checkpoint with ability to resume training.

    :param path: Path to save checkpoint.
    :param model: Model object.
    :param optimizer: Optimizer object.
    :param scheduler: Scheduler object.
    :param epoch: Epoch number.
    :param best_val_loss: Best validation loss.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, path)


def main():
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
    tqdm.write(f"Device: {device}")

    dataset = PennFudanDataset(root=args.data_root)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train

    train_subset, val_subset = random_split(dataset, (n_train, n_val), generator=Generator().manual_seed(42))

    train_dataset = PennFudanDataset(root=args.data_root, transforms=get_transform(train=True))
    val_dataset = PennFudanDataset(root=args.data_root, transforms=get_transform(train=False))

    trainset = Subset(train_dataset, train_subset.indices)
    valset = Subset(val_dataset, val_subset.indices)

    trainloader = DataLoader(trainset, args.batch_size, True, num_workers=args.num_workers, collate_fn=collate_fn)
    tqdm.write(f"Train dataset size: {len(trainset)}")

    valloader = DataLoader(valset, args.batch_size, False, num_workers=args.num_workers, collate_fn=collate_fn)
    tqdm.write(f"Val dataset size: {len(valset)}")

    net = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    net.to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device)

        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        tqdm.write(
            f"Resumed from {args.resume_from}: "
            f"last_epoch={checkpoint['epoch']}, "
            f"start_epoch={start_epoch}, "
            f"best_val_loss={best_val_loss:.4f}"
        )

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(mlflow_params)
        best_model_name = (
            f"{args.run_name.lower()}_penn_fudan_best_"
            f"batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
        )

        last_model_name = (
            f"{args.run_name.lower()}_penn_fudan_last_"
            f"batch{args.batch_size}_lr{args.lr}_momentum{args.momentum}.pt"
        )
        for epoch in range(start_epoch, args.epochs + 1):
            train_losses = train_loop(trainloader, net, optimizer, device, epoch)
            val_losses = val_loss_loop(valloader, net, device, epoch)

            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics(
                {
                    "train_loss_classifier": train_losses["loss_classifier"],
                    "train_loss_box_reg": train_losses["loss_box_reg"],
                    "train_loss_objectness": train_losses["loss_objectness"],
                    "train_loss_rpn_box_reg": train_losses["loss_rpn_box_reg"],
                    "train_total_loss": train_losses["total_loss"],
                    "val_loss_classifier": val_losses["loss_classifier"],
                    "val_loss_box_reg": val_losses["loss_box_reg"],
                    "val_loss_objectness": val_losses["loss_objectness"],
                    "val_loss_rpn_box_reg": val_losses["loss_rpn_box_reg"],
                    "val_total_loss": val_losses["total_loss"],
                    "lr": current_lr,
                },
                step=epoch,
            )

            scheduler.step()

            last_model_path = join(args.save_model_path, last_model_name)

            current_val_loss = val_losses["total_loss"]
            is_best = current_val_loss < best_val_loss

            if is_best:
                best_val_loss = current_val_loss

            # save last checkpoint every epoch for resume
            save_checkpoint(
                path=last_model_path,
                model=net,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
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
                    best_val_loss=best_val_loss,
                )
                mlflow.log_artifact(best_model_path)
                tqdm.write(f"New best model: epoch={epoch}, best_val_loss={best_val_loss:.4f}")
    tqdm.write(f"Finished Training")


if __name__ == "__main__":
    main()
