from argparse import ArgumentParser, Namespace
from os.path import join
from typing import Any

import mlflow
import torch
from torch import Generator, Tensor
from torch.nn import Module
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from eval import get_class_ids_from_model, collect_evaluation_data
from src.dataset.penn_fudan import PennFudanDataset
from src.evaluation.manual.detection_metrics import count_matches, calculate_metrics
from src.evaluation.manual.ranking_metrics import (
    build_ranked_matches,
    calculate_pr_curves,
    calculate_average_precisions,
    calculate_map,
)
from src.transforms import get_transform
from src.utils import collate_fn


def parse_args() -> Namespace:
    """Parse command-line arguments for the script.

    :return: Parsed command-line arguments.
    """
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

    # MLflow
    parser.add_argument("--mlflow_address", type=str, default="http://host.docker.internal:8081")
    parser.add_argument("--experiment_name", type=str, default="PennFudanPed")
    parser.add_argument("--run_name", type=str, default="fasterrcnn_resnet50_fpn")

    # Evaluation
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.5)

    return parser.parse_args()


def build_model(num_classes: int, use_pretrained_weights: bool) -> Module:
    """Build a Faster R-CNN model for the required number of classes.

    :param num_classes: Number of output classes, including the background class.
    :param use_pretrained_weights: Whether to initialize the model with pretrained COCO weights.
    :return: Faster R-CNN model with a replaced box predictor.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if use_pretrained_weights else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.score_thresh = 0.0001

    return model


def move_targets_to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    """Move tensor values inside target dictionaries to a device.

    :param targets: Target dictionaries returned by the dataset.
    :param device: Device where tensors should be moved.
    :return: Target dictionaries with tensor values moved to the selected device.
    """
    return [
        {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def train_loop(
        dataloader: DataLoader,
        model: Module,
        optimizer: Optimizer,
        device: torch.device,
        epoch: int,
) -> dict[str, float]:
    """Run one training epoch and return averaged losses.

    :param dataloader: DataLoader with training samples.
    :param model: Detection model to train.
    :param optimizer: Optimizer used to update model parameters.
    :param device: Device used for training.
    :param epoch: Current epoch number.
    :return: Averaged loss values for the epoch.
    """
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
        optimizer.zero_grad()

        images = [image.to(device) for image in data[0]]
        targets = move_targets_to_device(data[1], device)

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        total_loss.backward()
        optimizer.step()

        for name, value in loss_dict.items():
            loss_sums[name] += value.item()
        loss_sums["total_loss"] += total_loss.item()

        pbar.set_postfix(
            {
                "lr": f"{optimizer.param_groups[0]["lr"]:.5f}",
                "loss_cls": f"{loss_sums["loss_classifier"] / i:.3f}",
                "loss_box": f"{loss_sums["loss_box_reg"] / i:.3f}",
                "loss_obj": f"{loss_sums["loss_objectness"] / i:.3f}",
                "loss_rpn_box": f"{loss_sums["loss_rpn_box_reg"] / i:.3f}",
                "total_loss": f"{loss_sums["total_loss"] / i:.3f}",
            }
        )

    losses = {
        name: value / len(dataloader)
        for name, value in loss_sums.items()
    }

    return losses


def evaluate(
        model: Module,
        dataloader: DataLoader,
        device,
        class_ids: list[int],
        score_threshold: float,
        iou_threshold: float,
) -> tuple[dict[int, dict[str, float]], dict[int, float], float]:
    """Evaluate a detection model using fixed-threshold and ranking metrics.

    :param model: Detection model to evaluate.
    :param dataloader: DataLoader with validation samples.
    :param device: Device used for model inference.
    :param class_ids: Foreground class IDs.
    :param score_threshold: Score threshold used for precision, recall, and F1.
    :param iou_threshold: IoU threshold used for prediction matching.
    :return: Fixed-threshold metrics, per-class AP values, and mAP.
    """
    evaluation_data = collect_evaluation_data(model, dataloader, device)

    matches_count = count_matches(
        evaluation_data=evaluation_data,
        class_ids=class_ids,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    metrics = calculate_metrics(matches_count)

    ranked_matches = build_ranked_matches(
        evaluation_data=evaluation_data,
        class_ids=class_ids,
        iou_threshold=iou_threshold,
    )

    average_precisions = calculate_average_precisions(calculate_pr_curves(ranked_matches))
    map_value = calculate_map(average_precisions)

    return metrics, average_precisions, map_value


def save_checkpoint(
        path: str,
        model: Module,
        optimizer: Optimizer,
        scheduler: StepLR,
        epoch: int,
        best_val_ap50: float,
) -> None:
    """Save a checkpoint that can be used to resume training.

    :param path: Path where the checkpoint should be saved.
    :param model: Model whose weights should be saved.
    :param optimizer: Optimizer whose state should be saved.
    :param scheduler: Scheduler whose state should be saved.
    :param epoch: Last completed epoch number.
    :param best_val_ap50: Best validation AP@50 score observed so far.
    """
    checkpoint = {
        "epoch": epoch,
        "best_val_ap50": best_val_ap50,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)


def main():
    """Train Faster R-CNN on Penn-Fudan and log losses and metrics to MLflow."""
    args = parse_args()

    mlflow.set_tracking_uri(args.mlflow_address)
    mlflow.set_experiment(args.experiment_name)

    mlflow_params = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
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

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    tqdm.write(f"Train dataset size: {len(trainset)}")

    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    tqdm.write(f"Val dataset size: {len(valset)}")

    model = build_model(
        num_classes=2,
        use_pretrained_weights=args.resume_from is None,
    )
    model.to(device)
    class_ids = get_class_ids_from_model(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    start_epoch = 1
    best_val_ap50 = -float("inf")

    if args.resume_from is not None:
        checkpoint: dict[str, Any] = torch.load(args.resume_from, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_ap50 = checkpoint.get("best_val_ap50", best_val_ap50)

        tqdm.write(
            f"Resumed from {args.resume_from}: "
            f"last_epoch={checkpoint["epoch"]}, "
            f"start_epoch={start_epoch}, "
            f"best_val_ap50={best_val_ap50:.4f}"
        )

    best_model_name = (f"{model._get_name()}-{args.run_name.lower()}_penn_fudan_best.pt")
    last_model_name = (f"{model._get_name()}-{args.run_name.lower()}_penn_fudan_last.pt")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(mlflow_params)

        for epoch in range(start_epoch, args.epochs + 1):
            train_losses = train_loop(
                dataloader=trainloader,
                model=model,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
            metrics, average_precisions, mean_average_precisions = evaluate(
                dataloader=valloader,
                model=model,
                device=device,
                class_ids=class_ids,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            person_metrics = metrics[1]
            mlflow.log_metrics(
                {
                    "lr": current_lr,
                    "train_loss_objectness": train_losses["loss_objectness"],
                    "train_loss_rpn_box_reg": train_losses["loss_rpn_box_reg"],
                    "train_loss_classifier": train_losses["loss_classifier"],
                    "train_loss_box_reg": train_losses["loss_box_reg"],
                    "train_total_loss": train_losses["total_loss"],
                    "val_precision": person_metrics["precision"],
                    "val_recall": person_metrics["recall"],
                    "val_f1": person_metrics["f1"],
                    "val_ap50": average_precisions[1],
                    "val_map50": mean_average_precisions,
                },
                step=epoch,
            )

            scheduler.step()

            current_val_ap50 = average_precisions[1]
            is_best = current_val_ap50 > best_val_ap50

            if is_best:
                best_val_ap50 = current_val_ap50

            last_model_path = join(args.save_model_path, last_model_name)
            save_checkpoint(
                path=last_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_ap50=best_val_ap50,
            )
            mlflow.log_artifact(last_model_path)

            if is_best:
                best_model_path = join(args.save_model_path, best_model_name)
                save_checkpoint(
                    path=best_model_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_ap50=best_val_ap50,
                )
                mlflow.log_artifact(best_model_path)

                tqdm.write(f"New best model: epoch={epoch}, best_val_ap50={best_val_ap50:.4f}")

    tqdm.write("Finished Training")


if __name__ == "__main__":
    main()
