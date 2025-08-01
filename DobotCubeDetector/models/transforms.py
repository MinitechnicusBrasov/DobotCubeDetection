import torchvision
import torchvision.transforms.functional as F
import torch  # Import torch for tensor checks
import logging  # Import logging for potential debug messages
import torchvision.transforms.v2 as v2_transforms  # Renamed import

logger = logging.getLogger(__name__)


# Custom transform to conditionally apply RandomIoUCrop
class ConditionalRandomIoUCrop(v2_transforms.RandomIoUCrop):  # Use v2_transforms
    """
    A wrapper around RandomIoUCrop that only applies the transform if
    the input target contains non-empty bounding boxes.
    """

    def forward(self, img, target):
        # Check if there are any bounding boxes in the target
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            return super().forward(img, target)
        else:
            # If no boxes are present, log a debug message and return the original image and target
            # This prevents RandomIoUCrop from failing on images without annotations.
            logger.debug(
                "Skipping RandomIoUCrop as no bounding boxes are present in the target."
            )
            return img, target


def get_transform(train):
    """
    Defines the image transformations for training and validation/testing.

    Args:
        train (bool): If True, applies training-specific augmentations.
                      If False, applies only necessary transformations for evaluation.

    Returns:
        torchvision.transforms.v2.Compose or None: A composition of transformations.
                                                    Returns None if no transformations are needed.
    """
    transforms = []
    # The image is ALREADY a tensor when it comes out of the dataset's __getitem__
    # due to F.to_tensor. So, we don't need a ToTensor() transform here.

    if train:
        # Data augmentation transforms for training
        # These transforms are designed to work with both image tensors and target dictionaries
        # (containing boxes, labels, masks) from torchvision.transforms.v2.

        # Random horizontal flip: Flips the image horizontally with a given probability.
        transforms.append(v2_transforms.RandomHorizontalFlip(0.5))  # Use v2_transforms

        # Random photometric distortions: Adjusts brightness, contrast, saturation, and hue.
        # This helps the model generalize to different lighting conditions.
        transforms.append(
            v2_transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        )  # Use v2_transforms

        # Random rotation: Rotates the image by a random angle.
        # This helps the model become robust to objects appearing at different orientations.
        # The `expand=False` keeps the image size fixed, potentially cropping corners.
        # `interpolation` and `fill` handle how pixels are handled during rotation.
        transforms.append(
            v2_transforms.RandomRotation(  # Use v2_transforms
                degrees=15,  # Rotate by up to +/- 15 degrees
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                expand=False,
                fill=0,  # Fill any empty areas with black
            )
        )

        # Conditional Random IoU Crop: Only applies if bounding boxes are present.
        # transforms.append(
        #     ConditionalRandomIoUCrop(
        #         min_scale=0.3,  # Minimum scale of the cropped image relative to the original
        #         max_scale=1.0,  # Maximum scale (can be up to original size)
        #         min_aspect_ratio=0.5,  # Minimum aspect ratio of the cropped image
        #         max_aspect_ratio=2.0,  # Maximum aspect ratio
        #         sampler_options=[
        #             0.0,
        #             0.1,
        #             0.3,
        #             0.5,
        #             0.7,
        #             0.9,
        #             1.0,
        #         ],  # IoU thresholds to sample from
        #     )
        # )
        #
    # If there are any transforms, compose them. Otherwise, return None.
    if len(transforms) > 0:
        return v2_transforms.Compose(transforms)  # Use v2_transforms
    else:
        # For the validation/test set (when train=False), we return None
        # if no other specific transforms (like normalization, which is often
        # handled internally by the model's transform) are required.
        return None
