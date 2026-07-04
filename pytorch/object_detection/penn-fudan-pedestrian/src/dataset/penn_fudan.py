import os
from typing import Any, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F


class PennFudanDataset(Dataset):
    """Dataset for the Penn-Fudan pedestrian detection task.

    The dataset loads RGB images and instance segmentation masks, converts masks
    into bounding boxes, and returns samples in the format expected by
    torchvision detection models.
    """

    def __init__(self, root: str, transforms: Callable[[Tensor, dict[str, Any]], tuple[Tensor, dict[str, Any]]] = None):
        """Initialize the dataset.

        :param root: Path to the Penn-Fudan dataset root directory.
        :param transforms: Optional transforms applied to the image and target.
        """
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        """Load one image and its target annotations.

        :param idx: Index of the sample to load.
        :return: Image tensor and target dictionary with boxes, masks, labels,
            image id, areas, and crowd flags.
        """
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = decode_image(img_path)
        mask = decode_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        :return: Dataset size.
        """
        return len(self.imgs)
