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
from src.evaluation.manual.detection_metrics import count_matches, calculate_metrics
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
    parser.add_argument("--score_threshold", type=float, default=0.85)
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


def collect_evaluation_data(
        model: Module,
        dataloader: DataLoader,
        device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Run model inference and collect predictions and targets.

    :param model: Detection model to evaluate.
    :param dataloader: DataLoader with evaluation samples.
    :param device: Device used for model inference.
    :return: Per-image predictions and ground-truth annotations.
    """
    model.eval()

    evaluation_data: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for images, targets in tqdm(dataloader, desc="Collecting evaluation data"):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                evaluation_data.append({
                    "pred_boxes": output["boxes"].detach().cpu(),
                    "pred_scores": output["scores"].detach().cpu(),
                    "pred_labels": output["labels"].detach().cpu(),
                    "gt_boxes": target["boxes"].detach().cpu(),
                    "gt_labels": target["labels"].detach().cpu(),
                })

    return evaluation_data


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

    dataloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    tqdm.write(f"Eval dataset size: {len(valset)}")

    model = build_model(num_classes=2, checkpoint_path=args.load_model_path, device=device)
    class_ids = get_class_ids_from_model(model)

    evaluation_data = collect_evaluation_data(model=model, dataloader=dataloader, device=device)

    matches_count = count_matches(
        evaluation_data=evaluation_data,
        class_ids=class_ids,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
    )
    metrics = calculate_metrics(matches_count)

    for class_id, metrics_dict in metrics.items():
        tqdm.write(f"{class_id}, "
                   f"Precision: {metrics_dict['precision']:.3f}, "
                   f"Recall: {metrics_dict['recall']:.3f}, "
                   f"F1: {metrics_dict['f1']:.3f}")

    tqdm.write("Evaluating finished")


if __name__ == "__main__":
    main()
