import os

import torch
from torch.utils.data.dataset import Dataset
from torchvision.io import decode_image

class2idx = {
    "Car": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
}


class KITTIDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.img_dir = os.path.join(root, "image_2")
        self.label_dir = os.path.join(root, "label_2")

        self.img_files = sorted(os.listdir(self.img_dir))
        self.img_labels = self.__read_labels__(os.listdir(self.label_dir))
        self.transform = transform

    def __read_labels__(self, label_files: list[str]) -> dict[int, dict[str, torch.Tensor]]:
        img_labels = {}
        for label_file in sorted(label_files):
            boxes = []
            labels = []
            label_path = os.path.join(self.label_dir, label_file)
            with open(label_path, "r") as f:
                for line in f.readlines():
                    line = line.split()

                    class_name = line[0]
                    if class_name not in class2idx:
                        continue
                    xmin = float(line[4])
                    ymin = float(line[5])
                    xmax = float(line[6])
                    ymax = float(line[7])

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class2idx[class_name])

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            if boxes.numel() == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            area = (
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                if len(boxes) > 0
                else torch.zeros((0,), dtype=torch.float32)
            )
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

            img_id = os.path.splitext(label_file)[0]
            img_labels[int(img_id)] = {
                "boxes": boxes,
                "labels": labels,
                "area": area,
                "iscrowd": iscrowd,
            }
        return img_labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = decode_image(img_path)
        label = self.img_labels[idx]
        target = {
            "boxes": label["boxes"].clone(),
            "labels": label["labels"].clone(),
            "area": label["area"].clone(),
            "iscrowd": label["iscrowd"].clone(),
            "image_id": torch.tensor([idx]),
        }
        if self.transform:
            image = self.transform(image)
        return image, target
