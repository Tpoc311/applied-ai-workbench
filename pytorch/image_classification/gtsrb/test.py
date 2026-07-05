import os
from argparse import ArgumentParser, Namespace
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from tqdm import tqdm

from src.class_names import GTSRB_CLASS_NAMES
from src.transforms import get_val_transforms


def parse_args() -> Namespace:
    """Parse command-line arguments for the script.

    :return: Namespace containing parsed CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="artifacts/datasets/GTSRB")
    parser.add_argument("--model", type=str, help="Model architecture to use.", required=True)
    parser.add_argument("--models_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--reduce_cm_threshold", type=float, default=0.95)
    return parser.parse_args()


def val_loop(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: CrossEntropyLoss,
        device: torch.device,
) -> tuple[float, float, np.ndarray]:
    """Run the validation loop for the model.

    :param dataloader: DataLoader yielding validation batches (images, labels).
    :param model: Pre-trained neural network to evaluate.
    :param loss_fn: Loss function used to compute validation loss.
    :param device: PyTorch device (cpu/cuda) for tensor placement.
    :return: Tuple of (average_loss, accuracy) over the entire validation set.
    """
    model.eval()

    running_loss, correct, total = 0.0, 0.0, 0
    total_preds, total_labels = [], []

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=f"Testing")
    with torch.no_grad():
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)

            logits = model(inputs)
            running_loss += loss_fn(logits, labels).item()
            correct += (logits.argmax(1) == labels).type(torch.float).sum().item()
            total += labels.size(0)

            total_preds.append(torch.argmax(logits, dim=1))
            total_labels.append(labels)

            pbar.set_postfix({
                "loss": f"{running_loss / i:.3f}",
                "acc": f"{correct / total:.4f}",
            })

    y_pred = torch.cat(total_preds).cpu().numpy()
    y_true = torch.cat(total_labels).cpu().numpy()
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    return running_loss / len(dataloader), correct / total, cm_norm


def save_confusion_matrix(
        cm: np.ndarray,
        save_path: str,
        class_names: list[str],
        normalize: bool = False,
        include_values: bool = True,
        title: str | None = None,
        font_size: int = 18,
) -> None:
    """Save a confusion matrix as an image.

    :param cm: Confusion matrix to visualize.
    :param save_path: Path where the image should be saved.
    :param class_names: Class names used as axis labels.
    :param normalize: Whether the confusion matrix contains normalized values.
    :param include_values: Whether to draw values inside matrix cells.
    :param title: Optional plot title.
    :param font_size: Optional font size.
    """
    if title is None:
        title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"

    with plt.rc_context({"font.size": font_size}):
        fig, ax = plt.subplots(figsize=(30, 30))

        class_names = [f"{num}. {class_name}" for num, class_name in enumerate(class_names)]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        disp.plot(
            ax=ax,
            xticks_rotation="vertical",
            values_format=".2f" if normalize else "d",
            include_values=include_values,
            colorbar=True,
        )

        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


def get_hard_class_indices(cm: np.ndarray, threshold: float) -> np.ndarray:
    """Return class indices whose diagonal score is less than or equal to a threshold.

    If a raw confusion matrix is passed, the diagonal score is computed as the
    per-class ratio of correct predictions. If a normalized confusion matrix is
    passed, this is equivalent to checking the diagonal values directly.

    :param cm: Raw or normalized confusion matrix.
    :param threshold: Classes with a diagonal score greater than this value are removed.
    :return: Class indices that should be kept in the reduced confusion matrix.
    """
    class_support = cm.sum(axis=1)

    diagonal_ratio = np.divide(
        np.diag(cm),
        class_support,
        out=np.zeros_like(np.diag(cm), dtype=np.float64),
        where=class_support != 0,
    )

    hard_class_indices = np.where(diagonal_ratio <= threshold)[0]

    return hard_class_indices


def main():
    """Load the test dataset, evaluate saved models, and save confusion matrices."""
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Device: {device}")

    valset = GTSRB(root=args.data_root, split="test", transform=get_val_transforms())
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    tqdm.write(f"Test dataset size: {len(valset)}")

    with open("GTSRB_test_acc.txt", "w") as f:
        for model_name in sorted(os.listdir(args.models_dir)):
            if model_name.split(".")[-1] != "pt":
                continue

            net = models_dict[args.model]()
            net.fc = nn.Linear(net.fc.in_features, len(set([label for _, label in valset])))
            checkpoint = torch.load(join(args.models_dir, model_name), map_location=device)
            net.load_state_dict(checkpoint["model_state_dict"])
            net.to(device)

            tqdm.write(f"Testing model: {model_name}, epoch: {checkpoint["epoch"]}")
            val_loss, val_acc, cm_norm = val_loop(valloader, net, CrossEntropyLoss(), device)

            hard_class_indices = get_hard_class_indices(cm=cm_norm, threshold=args.reduce_cm_threshold)
            reduced_cm_norm = cm_norm[np.ix_(hard_class_indices, hard_class_indices)]
            reduced_class_names = [GTSRB_CLASS_NAMES[i] for i in hard_class_indices]
            model_stem = os.path.splitext(model_name)[0]

            save_confusion_matrix(
                cm=cm_norm,
                save_path=f"{model_stem}_cm_full_norm.png",
                class_names=GTSRB_CLASS_NAMES,
                normalize=True,
                include_values=False,
                title="Full Normalized Confusion Matrix",
            )

            save_confusion_matrix(
                cm=reduced_cm_norm,
                save_path=f"{model_stem}_cm_reduced_norm_thr_{args.reduce_cm_threshold}.png",
                class_names=reduced_class_names,
                normalize=True,
                include_values=True,
                title=f"Reduced Normalized Confusion Matrix: diag <= {args.reduce_cm_threshold}",
            )

            f.write(f"model: {model_name}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}.\n")
    tqdm.write(f"Finished Testing")


if __name__ == "__main__":
    models_dict = {
        "ResNet18": resnet18,
        "ResNet34": resnet34,
        "ResNet50": resnet50,
        "ResNet101": resnet101,
        "ResNet152": resnet152,
    }
    main()
