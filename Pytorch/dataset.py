import os
import json
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from eval_forward import eval_forward


class CubeDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading images and annotations
    from a COCO-formatted dataset for instance segmentation.
    """

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotation_file = os.path.join(root, "_annotations.coco.json")

        with open(self.annotation_file) as f:
            self.coco_data = json.load(f)

        self.images_info = self.coco_data["images"]
        self.annotations_info = self.coco_data["annotations"]

        self.image_to_anns = {}
        for ann in self.annotations_info:
            img_id = ann["image_id"]
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)

        self.cat_id_to_label = {
            cat["id"]: i + 1 for i, cat in enumerate(self.coco_data["categories"])
        }
        self.label_to_cat_name = {
            i + 1: cat["name"] for i, cat in enumerate(self.coco_data["categories"])
        }
        self.num_classes = len(self.coco_data["categories"])

    def __getitem__(self, idx):
        # Load image information
        image_info = self.images_info[idx]
        image_path = os.path.join(self.root, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # Get all annotations for this image
        image_id = image_info["id"]
        anns = self.image_to_anns.get(image_id, [])

        # Process annotations
        boxes, labels, masks = [], [], []
        for ann in anns:
            # Bounding Box
            bbox = ann["bbox"]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(self.cat_id_to_label[ann["category_id"]])

            # Segmentation Mask
            instance_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for poly in ann["segmentation"]:
                poly_points = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(instance_mask, [poly_points], 1)
            masks.append(instance_mask)

        # Convert annotations to torch Tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id_tensor = torch.tensor([image_id])
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        if masks:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            # Handle cases with no annotations
            masks = torch.empty((0, img_h, img_w), dtype=torch.uint8)
            area = torch.empty((0,), dtype=torch.float32)
            # Ensure boxes are empty but have the correct shape if there are no masks
            boxes = torch.empty((0, 4), dtype=torch.float32)

        # --- KEY CORRECTION ---
        # Explicitly convert the PIL Image to a PyTorch Tensor
        image = F.to_tensor(image)

        # Assemble the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id_tensor,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply optional augmentations
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images_info)


# --- 2. Model and Transforms ---
def get_transform(train):
    """Defines the image transformations (now only for augmentation)."""
    transforms = []
    # The image is ALREADY a tensor. We no longer need a conversion transform here.
    if train:
        # RandomHorizontalFlip works on tensors and targets together in v2
        transforms.append(torchvision.transforms.v2.RandomHorizontalFlip(0.5))

    # If there are any transforms, compose them. Otherwise, return None.
    if len(transforms) > 0:
        return torchvision.transforms.v2.Compose(transforms)
    else:
        # For the validation set (when train=False), we return None.
        return None


def get_model(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and modifies its classifier
     head for the number of classes in the custom dataset.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer_dim = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer_dim, num_classes
    )

    return model


def collate_fn(batch):
    """Custom collate function for the DataLoader."""
    return tuple(zip(*batch))
