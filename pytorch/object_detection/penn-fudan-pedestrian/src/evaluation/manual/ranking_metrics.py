import torch
from torch import Tensor
from torchvision.ops import box_iou


def build_ranked_matches(
        evaluation_data: list[dict[str, Tensor]],
        class_ids: list[int],
        iou_threshold: float,
) -> dict[int, dict[str, Tensor | int]]:
    """Match predictions with targets for each class across the dataset.

    Predictions are sorted by confidence score across the entire dataset.
    Each prediction is matched only with a ground-truth box from the same
    image and class.

    :param evaluation_data: Per-image predictions and ground-truth annotations.
    :param class_ids: Foreground class IDs to evaluate.
    :param iou_threshold: Minimum IoU required to count a prediction as TP.
    :return: Per-class scores, TP/FP flags, and total ground-truth count.
    """
    ranked_matches: dict[int, dict[str, Tensor | int]] = {}

    for class_id in class_ids:
        predictions: list[tuple[float, int, Tensor]] = []
        gt_boxes_by_image: list[Tensor] = []
        total_gt = 0

        for image_index, sample in enumerate(evaluation_data):
            pred_mask = sample["pred_labels"] == class_id
            gt_mask = sample["gt_labels"] == class_id

            pred_boxes = sample["pred_boxes"][pred_mask]
            pred_scores = sample["pred_scores"][pred_mask]
            gt_boxes = sample["gt_boxes"][gt_mask]

            gt_boxes_by_image.append(gt_boxes)
            total_gt += len(gt_boxes)

            for box, score in zip(pred_boxes, pred_scores):
                predictions.append((
                    score.item(),
                    image_index,
                    box,
                ))

        predictions.sort(key=lambda prediction: prediction[0], reverse=True)
        matched_gts = [torch.zeros(len(gt_boxes), dtype=torch.bool) for gt_boxes in gt_boxes_by_image]

        scores: list[float] = []
        tp_flags: list[int] = []
        fp_flags: list[int] = []

        for score, image_index, pred_box in predictions:
            scores.append(score)

            gt_boxes = gt_boxes_by_image[image_index]
            matched = matched_gts[image_index]

            if len(gt_boxes) == 0 or matched.all():
                tp_flags.append(0)
                fp_flags.append(1)
                continue

            iou_values = box_iou(
                pred_box.unsqueeze(0),
                gt_boxes,
            ).squeeze(0)

            # Already matched GT boxes cannot be used again.
            iou_values[matched] = -1.0

            best_iou, best_gt_index = torch.max(iou_values, dim=0)

            if best_iou >= iou_threshold:
                tp_flags.append(1)
                fp_flags.append(0)
                matched[best_gt_index] = True
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        ranked_matches[class_id] = {
            "scores": torch.tensor(scores, dtype=torch.float32),
            "tp_flags": torch.tensor(tp_flags, dtype=torch.float32),
            "fp_flags": torch.tensor(fp_flags, dtype=torch.float32),
            "total_gt": total_gt,
        }

    return ranked_matches


def calculate_pr_curves(ranked_matches: dict[int, dict[str, Tensor | int]]) -> dict[int, dict[str, Tensor]]:
    """Calculate a precision-recall curve for each class.

    :param ranked_matches: Per-class ranked prediction matching results.
    :return: Per-class scores, cumulative counts, precision, and recall.
    """
    pr_curves: dict[int, dict[str, Tensor]] = {}

    for class_id, matches in ranked_matches.items():
        scores = matches["scores"]
        tp_flags = matches["tp_flags"]
        fp_flags = matches["fp_flags"]
        total_gt = matches["total_gt"]

        cumulative_tp = torch.cumsum(tp_flags, dim=0)
        cumulative_fp = torch.cumsum(fp_flags, dim=0)

        precision_values = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall_values = cumulative_tp / total_gt

        pr_curves[class_id] = {
            "scores": scores,
            "tp_flags": tp_flags,
            "fp_flags": fp_flags,
            "cumulative_tp": cumulative_tp,
            "cumulative_fp": cumulative_fp,
            "precision": precision_values,
            "recall": recall_values,
        }

    return pr_curves


def calculate_average_precisions(pr_curves: dict[int, dict[str, Tensor]]) -> dict[int, float]:
    """Calculate non-interpolated AP for each class.

    AP is calculated as precision weighted by recall increments.

    :param pr_curves: Per-class precision-recall curves.
    :return: Average Precision value for each class.
    """
    average_precisions: dict[int, float] = {}

    for class_id, curve in pr_curves.items():
        precision_values = curve["precision"]
        recall_values = curve["recall"]

        if len(recall_values) == 0:
            average_precisions[class_id] = 0.0
            continue

        previous_recall = torch.cat([torch.tensor([0.0]), recall_values[:-1]])
        recall_deltas = recall_values - previous_recall
        average_precisions[class_id] = torch.sum(recall_deltas * precision_values).item()

    return average_precisions


def calculate_map(average_precisions: dict[int, float]) -> float:
    """Calculate mean Average Precision across classes.

    :param average_precisions: Per-class Average Precision values.
    :return: Mean Average Precision value.
    """
    return sum(average_precisions.values()) / len(average_precisions)
