import numpy as np
import cv2
import torch
from typing import Tuple, List, Dict, Optional
from torch import Tensor
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
import logging  # Import the logging module
from torchvision.models.detection.roi_heads import maskrcnn_loss  # Import maskrcnn_loss

# Get the logger for this module. It will inherit configuration from the root logger.
logger = logging.getLogger(__name__)


# --- Utility for calculating middle point of a bounding box ---
def calculate_middle(x1, y1, x2, y2):
    """
    Calculates the middle coordinates (centroid) of a bounding box.

    Args:
        x1 (int): X-coordinate of the top-left corner.
        y1 (int): Y-coordinate of the top-left corner.
        x2 (int): X-coordinate of the bottom-right corner.
        y2 (int): Y-coordinate of the bottom-right corner.

    Returns:
        tuple: A tuple (cX, cY) representing the center coordinates.
    """
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# --- Calibration Bounding Boxes (Example values, adjust as needed) ---
# These define regions in the camera feed where calibration points are expected.
# Format: [x_min, y_min, x_max, y_max]
BOX_1 = [20, 20, 120, 120]
BOX_2 = [500, 20, 600, 120]
BOX_3 = [20, 300, 120, 400]
BOX_4 = [500, 300, 600, 400]

# Calculate the middle points for the calibration boxes
MIDDLE_1 = calculate_middle(*BOX_1)
MIDDLE_2 = calculate_middle(*BOX_2)
MIDDLE_3 = calculate_middle(*BOX_3)
MIDDLE_4 = calculate_middle(*BOX_4)


# --- Calibration Functions ---
def show_calibration():
    """
    Displays a live camera feed with calibration boxes drawn on it.
    The user can press 'q' to quit this calibration preview.
    """
    cap = cv2.VideoCapture(2)  # Assuming camera index 2, adjust if needed

    if not cap.isOpened():
        logger.error("Could not open video stream for calibration.")  # Use logger.error
        return

    logger.info("Displaying calibration points. Press 'q' to quit.")  # Use logger.info
    while True:
        ret, frame = cap.read()

        if not ret:
            logger.warning(
                "Could not read frame during calibration."
            )  # Use logger.warning
            break  # Exit loop if frame cannot be read

        # Draw rectangles and text for each calibration position
        cv2.rectangle(frame, (BOX_1[0], BOX_1[1]), (BOX_1[2], BOX_1[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Position 1",
            (BOX_1[0], BOX_1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, MIDDLE_1, radius=10, color=(255, 255, 255), thickness=1)

        cv2.rectangle(frame, (BOX_2[0], BOX_2[1]), (BOX_2[2], BOX_2[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Position 2",
            (BOX_2[0], BOX_2[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, MIDDLE_2, radius=10, color=(255, 255, 255), thickness=1)

        cv2.rectangle(frame, (BOX_3[0], BOX_3[1]), (BOX_3[2], BOX_3[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Position 3",
            (BOX_3[0], BOX_3[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, MIDDLE_3, radius=10, color=(255, 255, 255), thickness=1)

        cv2.rectangle(frame, (BOX_4[0], BOX_4[1]), (BOX_4[2], BOX_4[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Position 4",
            (BOX_4[0], BOX_4[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, MIDDLE_4, radius=10, color=(255, 255, 255), thickness=1)

        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break  # Exit loop on 'q' press

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Calibration preview closed.")  # Use logger.info


def calibrate_and_transform():
    """
    Performs camera-to-robot coordinate calibration using homography.
    It displays calibration points, prompts the user to place the robot,
    and then calculates and returns the homography matrix.
    The homography matrix can be saved and reused.

    Returns:
        numpy.ndarray or None: The 3x3 homography matrix if successful, otherwise None.
    """
    # --- 1. DEFINE YOUR CALIBRATION POINTS ---
    # IMPORTANT: These robot_points should correspond to the real-world
    # coordinates (e.g., in millimeters) of the physical points
    # that align with the pixel_points shown on the camera feed.
    # The order of points must be the same in both arrays.

    # First, show the calibration UI to help the user align the robot.
    show_calibration()

    # (X, Y) coordinates from your robot's workspace in millimeters
    # These are example values and MUST be replaced with your actual measured robot coordinates.
    robot_points = np.array(
        [
            [353.1, 89.4],  # Point 1 (e.g., Top-Left corner of robot's viewable area)
            [355.8, -41.5],  # Point 2 (e.g., Top-Right)
            [281.6, -43.9],  # Point 3 (e.g., Bottom-Right)
            [277.0, 88.7],  # Point 4 (e.g., Bottom-Left)
        ],
        dtype="float32",
    )

    # (x, y) coordinates of the same points from your camera image in pixels.
    # These correspond to the MIDDLE_X points calculated from the BOX_X definitions.
    pixel_points = np.array(
        [
            [MIDDLE_1[0], MIDDLE_1[1]],  # Pixel coords of robot's Point 1
            [MIDDLE_2[0], MIDDLE_2[1]],  # Pixel coords of robot's Point 2
            [
                MIDDLE_4[0],
                MIDDLE_4[1],
            ],  # Pixel coords of robot's Point 3 (adjusted for correct mapping)
            [
                MIDDLE_3[0],
                MIDDLE_3[1],
            ],  # Pixel coords of robot's Point 4 (adjusted for correct mapping)
        ],
        dtype="float32",
    )

    # --- 2. CALCULATE THE HOMOGRAPHY MATRIX ---
    # This matrix 'h' is your calibration key. You can save it and reuse it.
    h, status = cv2.findHomography(pixel_points, robot_points)

    if h is None:
        logger.error(
            "Homography calculation failed. Check your points and ensure they are not collinear."
        )  # Use logger.error
        return None

    logger.info(
        f"✅ Homography Matrix Calculated. You can save and reuse this matrix:\n{h}"
    )  # Use logger.info

    # --- 3. TRANSFORM A NEW POINT (Example) ---
    # This section demonstrates how to use the calculated homography matrix.
    # In a real application, you would use this to convert detected object
    # pixel coordinates to robot coordinates.
    pixel_center_of_cube = np.array(
        [[[350, 250]]], dtype="float32"
    )  # Example pixel point

    # Use the matrix to transform the pixel point to a robot coordinate
    robot_coord = cv2.perspectiveTransform(pixel_center_of_cube, h)

    # The result is a nested array, so we extract the values.
    rx, ry = robot_coord[0][0]

    logger.info("\n--- Example Transformation Result ---")  # Use logger.info
    logger.info(
        f"Pixel Center: (px={pixel_center_of_cube[0][0][0]}, py={pixel_center_of_cube[0][0][1]})"
    )  # Use logger.info
    logger.info(
        f"Robot Coordinate: (RX={rx:.2f} mm, RY={ry:.2f} mm)"
    )  # Use logger.info

    return h


# --- Evaluation Forward Pass (for training and evaluation loops) ---
def eval_forward(model, images, targets):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Performs a forward pass through the model in evaluation mode,
    but also calculates losses if targets are provided.
    This function is adapted from torchvision's internal evaluation logic
    to allow loss calculation during validation.
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                logger.error(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    # Get proposals from RPN
    proposals, proposal_losses = model.rpn(images, features, targets)

    # ####################################################################
    # # MODIFICATION START: Replace manual ROI Heads logic with a single call #
    # ####################################################################

    # Temporarily set RoIHeads to training mode to get losses
    was_training = model.roi_heads.training
    model.roi_heads.training = True

    # Call the RoIHeads forward method directly.
    # It handles all internal logic, including mask projection and loss calculation.
    detections, detector_losses = model.roi_heads(
        features, proposals, images.image_sizes, targets
    )

    # Restore the original training state
    model.roi_heads.training = was_training

    # ##################################################################
    # # MODIFICATION END                                               #
    # ##################################################################

    # Post-process detections to map them back to the original image sizes
    detections = model.transform.postprocess(
        detections, images.image_sizes, original_image_sizes
    )

    # Combine all losses
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections
