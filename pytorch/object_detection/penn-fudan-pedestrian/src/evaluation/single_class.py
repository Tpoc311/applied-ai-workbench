import torch
from torch import Tensor, argsort
from torchvision.ops import box_iou


def filter_by_score(boxes: Tensor, scores: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    """Filter boxes and scores by a confidence threshold.

    :param boxes: Predicted bounding boxes.
    :param scores: Confidence scores for the predicted boxes.
    :param threshold: Minimum score required to keep a prediction.
    :return: Filtered bounding boxes and their corresponding scores.
    """
    mask = scores >= threshold
    return boxes[mask], scores[mask]


def sort_by_score(boxes: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]:
    """Sort boxes and scores by descending confidence score.

    :param boxes: Predicted bounding boxes.
    :param scores: Confidence scores for the predicted boxes.
    :return: Bounding boxes and scores sorted by score in descending order.
    """
    scores_indices = argsort(scores, descending=True)
    return boxes[scores_indices], scores[scores_indices]


def match_predictions_to_targets(
        pred_boxes: Tensor,
        pred_scores: Tensor,
        gt_boxes: Tensor,
        score_threshold: float,
        iou_threshold: float,
) -> tuple[int, int, int]:
    """Match predicted boxes to ground-truth boxes and compute TP, FP, and FN.

    Predictions are first filtered by score and sorted by descending score. Each prediction is matched to the
    unmatched ground-truth box with the highest IoU. A prediction is counted as a TP if its best IoU is greater
    than or equal to `iou_threshold`. Otherwise, it is counted as a false positive. Ground-truth boxes that
    remain unmatched are counted as false negatives.

    :param pred_boxes: Predicted bounding boxes.
    :param pred_scores: Confidence scores for the predicted boxes.
    :param gt_boxes: Ground-truth bounding boxes.
    :param score_threshold: Minimum confidence score required to keep a prediction.
    :param iou_threshold: Minimum IoU required to match a prediction to a ground-truth box.
    :return: Number of true positives, false positives, and false negatives.
    """
    pred_boxes, pred_scores = filter_by_score(pred_boxes, pred_scores, score_threshold)
    pred_boxes, pred_scores = sort_by_score(pred_boxes, pred_scores)

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    ious = box_iou(pred_boxes, gt_boxes)

    matched_gts = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
    tp, fp, fn = 0, 0, 0
    for pred_idx in range(len(pred_boxes)):
        iou_values = ious[pred_idx]

        # Do not allow one ground-truth box to be matched by multiple predictions.
        iou_values[matched_gts] = -1.0
        best_iou, best_gt_idx = torch.max(iou_values, dim=0)

        if best_iou >= iou_threshold and not matched_gts[best_gt_idx]:
            # The prediction matches the best still-unmatched ground-truth box.
            tp += 1
            matched_gts[best_gt_idx] = True
        else:
            # The prediction either has low IoU with all unmatched ground-truth boxes or duplicates an
            # already matched object.
            fp += 1

    fn = len(gt_boxes) - matched_gts.sum().item()

    return tp, fp, fn


def precision(tp: int, fp: int) -> float:
    """Compute precision from true positives and false positives.

    :param tp: Number of true positives.
    :param fp: Number of false positives.
    :return: Precision value. Returns ``0.0`` if ``tp + fp`` is zero.
    """
    return 0.0 if (tp + fp) == 0 else tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    """Compute recall from true positives and false negatives.

    :param tp: Number of true positives.
    :param fn: Number of false negatives.
    :return: Recall value. Returns ``0.0`` if ``tp + fn`` is zero.
    """
    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)


def f1(p: float, r: float) -> float:
    """Compute the F1 score from precision and recall.

    :param p: Precision value.
    :param r: Recall value.
    :return: F1 score. Returns ``0.0`` if ``p + r`` is zero.
    """
    return 0.0 if p + r == 0 else (2 * p * r) / (p + r)
