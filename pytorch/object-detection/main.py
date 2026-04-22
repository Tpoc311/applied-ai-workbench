import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.dataset import KITTIDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")

    for images, targets in progress_bar:
        # model expects list[Tensor] in [0, 1]
        images = [img.to(device).float() / 255.0 for img in images]

        targets = [
            {k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(data_loader)


def main():
    root = "artifacts/datasets/KITTI/object/2d-object/data_object_image_2/training"

    base_dataset = KITTIDataset(root=root)
    dataset_size = len(base_dataset)

    train_size = int(0.8 * dataset_size)
    indices = torch.randperm(
        dataset_size,
        generator=torch.Generator().manual_seed(42)
    ).tolist()

    train_indices = indices[:train_size]
    train_dataset = Subset(KITTIDataset(root=root), train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 4  # background + Car + Pedestrian + Cyclist
    model = create_model(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # sanity check: one forward pass
    images, targets = next(iter(train_loader))
    images = [img.to(device).float() / 255.0 for img in images]
    targets = [
        {k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()}
        for t in targets
    ]

    model.train()
    loss_dict = model(images, targets)
    print("Sanity check loss_dict:", loss_dict)

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

    print("Done")


if __name__ == "__main__":
    main()
