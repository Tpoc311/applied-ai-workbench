from argparse import ArgumentParser, Namespace
from typing import Any

import torch
from torch import Generator
from torch.nn import Module
from torch.utils.data import random_split, Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.dataset.penn_fudan import PennFudanDataset
from src.evaluation.manual_detection_metrics import match_predictions_to_targets_multiclass, precision, recall, f1
from src.transforms import get_transform
from src.utils import collate_fn


def parse_args() -> Namespace:
    """Parse command-line arguments for the script.

    :return: Namespace containing parsed CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="artifacts/datasets/PennFudanPed")
    parser.add_argument("--load_model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--score_threshold", type=float, default=0.9)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    return parser.parse_args()


def build_model(num_classes: int, checkpoint_path: str, device: torch.device) -> Module:
    """Build a Faster R-CNN model and load checkpoint weights.

    :param num_classes: Number of output classes, including the background class.
    :param checkpoint_path: Path to the saved checkpoint.
    :param device: Device used to load the checkpoint and run the model.
    :return: Model with loaded weights.
    """
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    return model


def evaluate(
        model: Module,
        dataloader: DataLoader,
        class_ids: list[int],
        device: torch.device,
        score_threshold: float,
        iou_threshold: float
) -> dict[int, dict[str, float]]:
    """Evaluate a detection model and accumulate TP, FP, and FN.

    :param model: Detection model to evaluate.
    :param dataloader: DataLoader with validation or test samples.
    :param class_ids: Class IDs list for hte model.
    :param device: Device used for model inference.
    :param score_threshold: Minimum confidence score required to keep a prediction.
    :param iou_threshold: Minimum IoU required to match a prediction to a ground-truth box.
    :return: Total numbers of true positives, false positives, and false negatives.
    """
    model.eval()

    matches_count: dict[int, dict[str, int]] = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in class_ids
    }
    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = [image.to(device) for image in inputs]

            outputs = model(inputs)

            for target, output in zip(targets, outputs):
                matches = match_predictions_to_targets_multiclass(
                    pred_boxes=output["boxes"].detach().cpu(),
                    pred_scores=output["scores"].detach().cpu(),
                    pred_labels=output["labels"].detach().cpu(),
                    gt_labels=target["labels"].detach().cpu(),
                    gt_boxes=target["boxes"].detach().cpu(),
                    class_ids=class_ids,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )

                for class_id in class_ids:
                    matches_count[class_id]["tp"] += matches[class_id]["tp"]
                    matches_count[class_id]["fp"] += matches[class_id]["fp"]
                    matches_count[class_id]["fn"] += matches[class_id]["fn"]

    return calculate_metrics(matches_count, class_ids)


def calculate_metrics(matches_count: dict[int, dict[str, int]], class_ids: list[int]) -> dict[int, dict[str, float]]:
    """Calculate detection metrics for each class.

    For each class ID, the function reads TP, FP, and FN counts and calculates
    precision, recall, and F1 score.

    :param matches_count: Per-class matching results with TP, FP, and FN counts.
    :param class_ids: Class IDs to calculate metrics for.
    :return: Per-class precision, recall, and F1 score values.
    """
    metrics: dict[int, dict[str, float]] = {
        class_id: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for class_id in class_ids
    }
    for class_id in class_ids:
        tp = matches_count[class_id]["tp"]
        fp = matches_count[class_id]["fp"]
        fn = matches_count[class_id]["fn"]

        metrics[class_id]["precision"] = precision(tp, fp)
        metrics[class_id]["recall"] = recall(tp, fn)
        metrics[class_id]["f1"] = f1(metrics[class_id]["precision"], metrics[class_id]["recall"])

    return metrics


def get_class_ids_from_model(model: Module) -> list[int]:
    """Get foreground class ids from a torchvision detection model.

    :param model: Detection model with a Fast R-CNN box predictor.
    :return: Foreground class ids excluding the background class.
    """
    num_classes = model.roi_heads.box_predictor.cls_score.out_features
    return list(range(1, num_classes))


def main():
    """Run model evaluation on the Penn-Fudan validation split."""
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Device: {device}")

    dataset = PennFudanDataset(root=args.data_root)

    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train

    _, val_subset = random_split(dataset, (n_train, n_val), generator=Generator().manual_seed(42))
    val_dataset = PennFudanDataset(root=args.data_root, transforms=get_transform(train=False))
    valset = Subset(val_dataset, val_subset.indices)

    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    tqdm.write(f"Test dataset size: {len(valset)}")

    model = build_model(num_classes=2, checkpoint_path=args.load_model_path, device=device)
    class_ids = get_class_ids_from_model(model)

    metrics = evaluate(
        model=model,
        dataloader=valloader,
        class_ids=class_ids,
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold
    )

    for class_id in class_ids:
        tqdm.write(f"{class_id}, "
                   f"Precision: {metrics[class_id]['precision']:.3f}, "
                   f"Recall: {metrics[class_id]['recall']:.3f}, "
                   f"F1: {metrics[class_id]['f1']:.3f}")

    tqdm.write("Finished Testing")


if __name__ == "__main__":
    main()
