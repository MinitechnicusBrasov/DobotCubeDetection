import os
import json
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from eval_forward import eval_forward

class CubeDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading images and annotations
    from a COCO-formatted dataset.
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotation_file = os.path.join(root, "_annotations.coco.json")

        # Load annotations
        with open(self.annotation_file) as f:
            self.coco_data = json.load(f)

        self.images_info = self.coco_data['images']
        self.annotations_info = self.coco_data['annotations']

        # Create a mapping from image_id to all its annotations
        self.image_to_anns = {}
        for ann in self.annotations_info:
            img_id = ann['image_id']
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)

        # Create a mapping from category_id to a contiguous range [0, num_classes-1]
        self.cat_id_to_label = {cat['id']: i + 1 for i, cat in enumerate(self.coco_data['categories'])}
        self.label_to_cat_name = {i + 1: cat['name'] for i, cat in enumerate(self.coco_data['categories'])}
        self.num_classes = len(self.coco_data['categories'])

    def __getitem__(self, idx):
        # Get image info for the given index
        image_info = self.images_info[idx]
        image_path = os.path.join(self.root, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        image_id = image_info['id']

    # Get all annotations for this image
        anns = self.image_to_anns.get(image_id, [])

        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(self.cat_id_to_label[ann['category_id']])

    # --- START OF THE FIX ---
    # Explicitly convert the PIL Image to a PyTorch Tensor.
    # This function also correctly scales pixel values to the [0.0, 1.0] range.
        image = F.to_tensor(image)
    # --- END OF THE FIX ---

    # Convert annotation data to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # Handle cases with no bounding boxes
        if boxes.shape[0] == 0:
        # Create empty tensors with the correct shape
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id_tensor = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

    # Create the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id_tensor, # Use the tensor version
            "area": area,
            "iscrowd": iscrowd
        }

    # Apply subsequent transforms (like augmentations) if they exist
    # These transforms now receive a Tensor, not a PIL image.
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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    # num_classes includes the background class, so it's your_classes + 1
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def collate_fn(batch):
    """Custom collate function for the DataLoader."""
    return tuple(zip(*batch))
