import os
import torch
from torch.utils.data import DataLoader
import logging  # Import the logging module
import cv2  # Required for the optional inference visualization block
import numpy as np  # Required for the optional inference visualization block

# Import components from the restructured project
from DobotCubeDetector.models.mask_rcnn_model import get_model
from DobotCubeDetector.models.transforms import get_transform
from DobotCubeDetector.training.dataset import CubeDataset, collate_fn
from DobotCubeDetector.inference.utils import (
    eval_forward,
)  # eval_forward is used for validation loss

# Get the logger for this module. It will inherit configuration from the root logger.
logger = logging.getLogger(__name__)


# --- Training and Evaluation Functions ---
def train_one_epoch(model, optimizer, data_loader, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The object detection model.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        device (torch.device): The device (CPU or GPU) to run training on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()  # Set model to training mode
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move images and targets to the specified device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # The model returns a dictionary of losses in training mode
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())  # Sum all individual losses
        loss_value = losses.item()  # Get the scalar loss value
        total_loss += loss_value

        # Backpropagation and optimization step
        optimizer.zero_grad()  # Clear previous gradients
        losses.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        # Log batch-wise loss (optional, can be verbose)
        # logger.debug(f"Batch {batch_idx+1}/{len(data_loader)} Loss: {loss_value:.4f}")

    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch training loss: {avg_loss:.4f}")  # Use logger.info
    return avg_loss


@torch.no_grad()  # Decorator to disable gradient calculation for evaluation
def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The object detection model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) to run evaluation on.

    Returns:
        float: The average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    for images, targets in data_loader:
        # Move images and targets to the specified device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # In evaluation, we use eval_forward to get validation loss
        # and also potential detections (though detections are not used for loss here).
        loss_dict, _ = eval_forward(model, images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        total_loss += loss_value

    avg_loss = total_loss / len(data_loader)
    logger.info(f"Validation loss: {avg_loss:.4f}")  # Use logger.info
    return avg_loss


# --- Main Execution Block for Training ---
# if __name__ == "__main__":
#     # Import logging setup from utils
#     from DobotCubeDetector.utils.logging_config import setup_logging
#
#     # Configure logging for standalone execution of this script
#     setup_logging(log_level=logging.INFO)
#
#     # --- Configuration ---
#     # IMPORTANT: Update DATASET_ROOT to point to your actual dataset location.
#     # For example, if your 'dataset' folder is directly inside 'DobotCubeDetector/',
#     # then DATASET_ROOT should be "dataset".
#     DATASET_ROOT = "./Pytorch/dataset"  # Assuming original Pytorch/dataset structure
#     TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
#     VALID_DIR = os.path.join(DATASET_ROOT, "valid")
#     TEST_DIR = os.path.join(
#         DATASET_ROOT, "test"
#     )  # Not directly used in training, but good to define
#     MODEL_SAVE_PATH = (
#         "cube_detector_model.pth"  # Path to save the trained model weights
#     )
#
#     NUM_EPOCHS = 70
#     BATCH_SIZE = 5
#     LEARNING_RATE = 0.005
#
#     # --- Setup ---
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     logger.info(f"Using device: {device}")  # Use logger.info
#
#     # Initialize datasets
#     # get_transform is imported from DobotCubeDetector.models.transforms
#     dataset_train = CubeDataset(root=TRAIN_DIR, transforms=get_transform(train=True))
#     dataset_valid = CubeDataset(root=VALID_DIR, transforms=get_transform(train=False))
#
#     # The number of classes is the number of cube colors + 1 for the background
#     # This assumes CubeDataset correctly parses the COCO annotations to determine num_classes.
#     num_classes = dataset_train.num_classes + 1
#
#     # Initialize data loaders
#     data_loader_train = DataLoader(
#         dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
#     )
#     data_loader_valid = DataLoader(
#         dataset_valid,
#         batch_size=1,
#         shuffle=False,
#         collate_fn=collate_fn,  # Batch size 1 for validation is common
#     )
#
#     # --- Model Training ---
#     # get_model is imported from DobotCubeDetector.models.rcnn_model
#     model = get_model(num_classes).to(device)
#
#     # Optimizer setup
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(
#         params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005
#     )
#
#     logger.info("\n--- Starting Training ---")  # Use logger.info
#     for epoch in range(NUM_EPOCHS):
#         logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")  # Use logger.info
#         train_one_epoch(model, optimizer, data_loader_train, device)
#         evaluate(model, data_loader_valid, device)  # Evaluate after each epoch
#
#     logger.info("\n--- Training Finished ---")  # Use logger.info
#     # Save the trained model's state dictionary
#     torch.save(model.state_dict(), MODEL_SAVE_PATH)
#     logger.info(f"Model saved to {MODEL_SAVE_PATH}")  # Use logger.info
#
#     # --- Optional: Inference and Visualization after Training ---
#     # This section is kept from the original train_cubes.py for convenience
#     # to quickly check a test image after training.
#     # For live inference, use inference/predictor.py.
#     logger.info(
#         "\n--- Running Inference on a Test Image (Post-Training Check) ---"
#     )  # Use logger.info
#
#     # Load the trained model for inference
#     inference_model = get_model(num_classes)
#     inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
#     inference_model.to(device)
#     inference_model.eval()  # Set to evaluation mode
#
#     # Load a test image and its annotations
#     dataset_test = CubeDataset(root=TEST_DIR, transforms=get_transform(train=False))
#
#     if len(dataset_test) == 0:
#         logger.warning(
#             "Test dataset is empty. Skipping inference visualization."
#         )  # Use logger.warning
#     else:
#         img, _ = dataset_test[1]  # Take the second test image (index 1)
#         label_map = dataset_test.label_to_cat_name
#
#         with torch.no_grad():
#             prediction = inference_model([img.to(device)])
#             logger.debug(
#                 f"Raw prediction output: {prediction}"
#             )  # Use logger.debug for verbose output
#
#         # Process the output for visualization
#         # Convert PyTorch tensor image back to NumPy for OpenCV
#         image_np = img.mul(255).permute(1, 2, 0).byte().numpy()
#         image_np = cv2.cvtColor(
#             image_np, cv2.COLOR_RGB2BGR
#         )  # Convert RGB to BGR for OpenCV
#
#         for box, label, score, mask in zip(
#             prediction[0]["boxes"],
#             prediction[0]["labels"],
#             prediction[0]["scores"],
#             prediction[0]["masks"],
#         ):
#             if score > 0.8:  # Confidence threshold for visualization
#                 logger.info(
#                     f"Confidence reached for label {label_map.get(label.item(), 'Unknown')}: {score:.2f}"
#                 )  # Use logger.info
#                 box = box.cpu().numpy().astype(int)
#                 label_name = label_map.get(label.item(), "Unknown")
#
#                 # Draw bounding box
#                 cv2.rectangle(
#                     image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
#                 )
#
#                 # Put label and score text
#                 label_text = f"{label_name}: {score:.2f}"
#                 cv2.putText(
#                     image_np,
#                     label_text,
#                     (box[0], box[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 0),
#                     2,
#                 )
#
#                 # Draw segmentation mask (if available and confidence is high)
#                 mask = mask.squeeze().cpu().numpy()
#                 binary_mask = (mask > 0.5).astype(
#                     np.uint8
#                 ) * 255  # Convert float mask to binary image
#                 contours, _ = cv2.findContours(
#                     binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#                 )
#
#                 if len(contours) > 0:
#                     # Get the minimum area rotated rectangle for the largest contour
#                     largest_contour = max(contours, key=cv2.contourArea)
#                     rect = cv2.minAreaRect(largest_contour)
#
#                     # The 'rect' object contains: (center, (width, height), angle)
#                     box_points = cv2.boxPoints(rect)
#                     box_points = np.intp(box_points)  # Convert points to integer
#
#                     # Extract the angle
#                     angle = rect[2]
#
#                     # Draw the rotated rectangle and the angle on the frame
#                     cv2.drawContours(
#                         image_np, [box_points], 0, (255, 0, 255), 2
#                     )  # Draw rotated box in magenta
#                     center_x, center_y = int(rect[0][0]), int(rect[0][1])
#                     cv2.putText(
#                         image_np,
#                         f"Angle: {angle:.1f}",
#                         (center_x - 40, center_y),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (255, 0, 255),
#                         2,
#                     )
#
#         # Save or display the result
#         cv2.imwrite("test_output.jpg", image_np)
#         logger.info("Inference result saved to test_output.jpg")  # Use logger.info
#         # To display in a window (if you have a GUI environment):
#         cv2.imshow("Inference Result", image_np)
#         cv2.waitKey(0)  # Wait indefinitely until a key is pressed
#         cv2.destroyAllWindows()
