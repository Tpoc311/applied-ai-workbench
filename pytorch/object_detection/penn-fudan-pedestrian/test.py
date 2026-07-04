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
from src.evaluation.single_class import match_predictions_to_targets
from src.evaluation.single_class import precision, recall, f1
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


def val_loop(model, dataloader, device, score_threshold=0.5, iou_threshold=0.5):
    """Evaluate a detection model and accumulate TP, FP, and FN.

    :param model: Detection model to evaluate.
    :param dataloader: DataLoader with validation or test samples.
    :param device: Device used for model inference.
    :param score_threshold: Minimum confidence score required to keep a prediction.
    :param iou_threshold: Minimum IoU required to match a prediction to a ground-truth box.
    :return: Total numbers of true positives, false positives, and false negatives.
    """
    model.eval()

    tp_total, fp_total, fn_total = 0, 0, 0

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = [image.to(device) for image in inputs]

            outputs = model(inputs)

            for target, output in zip(targets, outputs):
                tp, fp, fn = match_predictions_to_targets(
                    pred_boxes=output["boxes"].detach().cpu(),
                    pred_scores=output["scores"].detach().cpu(),
                    gt_boxes=target["boxes"].detach().cpu(),
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )

                tp_total += tp
                fp_total += fp
                fn_total += fn

    return tp_total, fp_total, fn_total


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

    tp_total, fp_total, fn_total = val_loop(model=model, dataloader=valloader, device=device)
    p = precision(tp_total, fp_total)
    r = recall(tp_total, fn_total)
    f = f1(p, r)

    tqdm.write(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
    tqdm.write("Finished Testing")


if __name__ == "__main__":
    main()
