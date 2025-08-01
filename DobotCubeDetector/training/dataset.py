import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import logging  # Import the logging module

# Get the logger for this module. It will inherit configuration from the root logger.
logger = logging.getLogger(__name__)


class CubeDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading images and annotations
    from a COCO-formatted dataset for instance segmentation.
    """

    def __init__(self, root, transforms=None):
        """
        Initializes the CubeDataset.

        Args:
            root (str): The root directory of the dataset (e.g., 'path/to/train').
                        This directory should contain image files and '_annotations.coco.json'.
            transforms (callable, optional): A function/transform that takes in a PIL image
                                             and a target dictionary, and returns transformed versions.
                                             (e.g., from DobotCubeDetector.models.transforms.get_transform)
        """
        self.root = root
        self.transforms = transforms
        self.annotation_file = os.path.join(root, "_annotations.coco.json")

        if not os.path.exists(self.annotation_file):
            logger.error(f"Annotation file not found at: {self.annotation_file}")
            raise FileNotFoundError(
                f"Annotation file not found at: {self.annotation_file}"
            )

        try:
            with open(self.annotation_file) as f:
                self.coco_data = json.load(f)
            logger.info(f"Loaded annotations from: {self.annotation_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.annotation_file}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while loading {self.annotation_file}: {e}"
            )
            raise

        self.images_info = self.coco_data["images"]
        self.annotations_info = self.coco_data["annotations"]

        # Create a mapping from image ID to its annotations for quicker lookup
        self.image_to_anns = {}
        for ann in self.annotations_info:
            img_id = ann["image_id"]
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)

        # Create mappings for category IDs to labels and vice-versa
        # Category IDs from COCO format are usually 1-indexed or arbitrary.
        # We map them to 0-indexed (or 1-indexed for model output if background is 0).
        # Here, we map to 1-indexed labels for consistency with typical object detection models
        # where label 0 is background.
        self.cat_id_to_label = {
            cat["id"]: i + 1 for i, cat in enumerate(self.coco_data["categories"])
        }
        self.label_to_cat_name = {
            i + 1: cat["name"] for i, cat in enumerate(self.coco_data["categories"])
        }
        # The number of classes excluding background (actual object classes)
        self.num_classes = len(self.coco_data["categories"])
        logger.info(
            f"Dataset initialized with {len(self.images_info)} images and {self.num_classes} object classes."
        )

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding annotations.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing:
                   - image (torch.Tensor): The image tensor.
                   - target (dict): A dictionary containing bounding boxes, labels, masks, etc.
        """
        # Load image information
        image_info = self.images_info[idx]
        image_path = os.path.join(self.root, image_info["file_name"])

        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}. Skipping this image.")
            # Return dummy data or raise an error, depending on desired behavior
            # For now, we'll return empty data and let the collate_fn handle it if needed
            # or rely on DataLoader's error handling for bad samples.
            # A more robust solution might involve filtering out bad image_info upfront.
            return torch.empty(3, 1, 1), {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "masks": torch.empty((0, 1, 1), dtype=torch.uint8),
                "image_id": torch.tensor([image_info["id"]]),
                "area": torch.empty((0,)),
                "iscrowd": torch.empty((0,), dtype=torch.int64),
            }

        image = Image.open(image_path).convert("RGB")  # Load image using PIL

        img_w, img_h = image.size

        # Get all annotations for this image
        image_id = image_info["id"]
        anns = self.image_to_anns.get(image_id, [])

        # Process annotations
        boxes, labels, masks = [], [], []
        for ann in anns:
            # Bounding Box: Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
            bbox = ann["bbox"]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(self.cat_id_to_label[ann["category_id"]])

            # Segmentation Mask: Create a binary mask from polygons
            instance_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # COCO segmentation can be RLE or polygon. This assumes polygon.
            for poly in ann["segmentation"]:
                # Reshape polygon points for cv2.fillPoly
                poly_points = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(
                    instance_mask, [poly_points], 1
                )  # Fill with 1 for object, 0 for background
            masks.append(instance_mask)

        # Convert annotations to torch Tensors
        # Boxes should be float32
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels should be int64
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Image ID is usually a single tensor
        image_id_tensor = torch.tensor([image_id])
        # iscrowd indicates if an annotation is a crowd annotation (0 for individual objects)
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        if masks:
            # Masks should be uint8
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            # Calculate area for each bounding box
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            # Handle cases with no annotations for an image
            masks = torch.empty((0, img_h, img_w), dtype=torch.uint8)
            area = torch.empty((0,), dtype=torch.float32)
            boxes = torch.empty(
                (0, 4), dtype=torch.float32
            )  # Ensure boxes have correct shape even if empty
            logger.warning(
                f"Image {image_info['file_name']} (ID: {image_id}) has no annotations."
            )

        # --- KEY CORRECTION ---
        # Explicitly convert the PIL Image to a PyTorch Tensor here.
        # This ensures that subsequent transforms (if any) operate on tensors.
        image = F.to_tensor(image)

        # Assemble the target dictionary required by torchvision models
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id_tensor,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply optional augmentations/transforms
        if self.transforms is not None:
            # Transforms should accept (image, target) and return (transformed_image, transformed_target)
            try:
                image, target = self.transforms(image, target)
            except Exception as e:
                logger.error(
                    f"Error applying transforms to image {image_info['file_name']} (ID: {image_id}): {e}"
                )
                # Re-raise the exception or handle gracefully, e.g., return original image/target
                raise

        return image, target

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images_info)


def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    This function is necessary when working with object detection datasets
    where each item in a batch can have a different number of annotations.
    """
    # Filter out any None values if __getitem__ returns None for problematic samples
    # (though current __getitem__ returns empty tensors instead of None)
    # batch = [item for item in batch if item is not None]
    if not batch:  # Handle case where batch might become empty after filtering
        logger.warning("Collate function received an empty batch.")
        return [], []  # Return empty lists for images and targets

    return tuple(zip(*batch))


# This block is for testing dataset functionality directly.
# It will only run if this file is executed as the main script.
if __name__ == "__main__":
    from DobotCubeDetector.utils.logging_config import setup_logging
    from DobotCubeDetector.models.transforms import get_transform
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    setup_logging(
        log_level=logging.DEBUG
    )  # Set to DEBUG for detailed dataset loading info

    # IMPORTANT: Adjust this path to a valid dataset for testing
    TEST_DATASET_ROOT = (
        "./Pytorch/dataset/train"  # Example path, replace with your actual dataset
    )

    logger.info(f"Attempting to load CubeDataset from: {TEST_DATASET_ROOT}")

    try:
        dataset = CubeDataset(
            root=TEST_DATASET_ROOT, transforms=get_transform(train=True)
        )
        logger.info(f"Successfully loaded dataset with {len(dataset)} images.")

        if len(dataset) > 0:
            # Try to load and visualize a few samples
            for i in range(min(5, len(dataset))):  # Load up to 5 samples
                logger.info(f"Loading sample {i}...")
                image, target = dataset[i]
                logger.debug(
                    f"Sample {i} - Image shape: {image.shape}, Target keys: {target.keys()}"
                )

                # Visualize the image and bounding boxes
                fig, ax = plt.subplots(1)
                # Convert tensor to numpy array for plotting
                img_to_plot = image.permute(1, 2, 0).cpu().numpy()
                ax.imshow(img_to_plot)

                for box, label in zip(target["boxes"], target["labels"]):
                    x_min, y_min, x_max, y_max = box.cpu().numpy()
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    label_name = dataset.label_to_cat_name.get(label.item(), "Unknown")
                    ax.text(
                        x_min,
                        y_min - 5,
                        label_name,
                        color="red",
                        fontsize=8,
                        bbox=dict(
                            facecolor="white", alpha=0.7, edgecolor="none", pad=0.5
                        ),
                    )

                plt.title(f"Sample {i} - Image ID: {target['image_id'].item()}")
                plt.axis("off")
                plt.show()
                logger.info(f"Displayed sample {i}.")
        else:
            logger.warning("Dataset is empty, no samples to display.")

    except FileNotFoundError as e:
        logger.critical(
            f"Dataset initialization failed: {e}. Please check DATASET_ROOT path."
        )
    except Exception as e:
        logger.critical(
            f"An error occurred during dataset loading or visualization: {e}",
            exc_info=True,
        )
