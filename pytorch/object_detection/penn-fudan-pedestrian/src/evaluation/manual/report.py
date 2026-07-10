from pathlib import Path

import matplotlib.pyplot as plt
from torch import Tensor


def save_pr_curves(
        pr_curves: dict[int, dict[str, Tensor]],
        average_precisions: dict[int, float],
        output_dir: Path,
        iou_threshold: float,
) -> None:
    """Save a precision-recall curve for each class.

    :param pr_curves: Per-class precision-recall curves.
    :param average_precisions: Per-class Average Precision values.
    :param output_dir: Directory used to save the plots.
    :param iou_threshold: IoU threshold used to build the curves.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_id, curve in pr_curves.items():
        precision_values = curve["precision"].numpy()
        recall_values = curve["recall"].numpy()
        ap = average_precisions[class_id]

        plt.figure(figsize=(8, 6))
        plt.step(recall_values, precision_values, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve | Class {class_id} | IoU={iou_threshold:.2f} | AP={ap:.4f}")
        plt.xlim(0.0, 1.05)
        plt.ylim(0.0, 1.05)
        plt.grid()
        plt.tight_layout()

        output_path = output_dir / f"pr_curve_class_{class_id}_iou_{iou_threshold:.2f}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
