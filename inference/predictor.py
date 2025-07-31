import cv2
import numpy as np
import os
import datetime
import torch
import torchvision.transforms.functional as F
import logging  # Import the logging module

# Import components from the restructured project
from DobotCubeDetector.models.mask_rcnn_model import get_model
from DobotCubeDetector.models.transforms import get_transform
from DobotCubeDetector.training.dataset import (
    CubeDataset,
    collate_fn,
)  # CubeDataset is needed for label mapping
from DobotCubeDetector.inference.utils import calculate_middle, calibrate_and_transform

# Get the logger for this module. It will inherit configuration from the root logger.
logger = logging.getLogger(__name__)

# --- Configuration for OpenCV-based detection ---
# Directory to save captured images for dataset creation
DATA_CAPTURE_DIR = "data/raw_image_captured"
if not os.path.exists(DATA_CAPTURE_DIR):
    os.makedirs(DATA_CAPTURE_DIR)
    logger.info(f"Created data capture directory: {DATA_CAPTURE_DIR}")

# Define color ranges in HSV. These values might need adjustment based on your lighting
# and the specific colors of your cubes.
# Format: (Hue_min, Saturation_min, Value_min), (Hue_max, Saturation_max, Value_max)
COLOR_RANGES = {
    "red": ([0, 100, 100], [10, 255, 255]),
    "red_upper": ([170, 100, 100], [180, 255, 255]),  # Red wraps around in HSV
    "green": ([40, 40, 40], [80, 255, 255]),
    "blue": ([100, 100, 100], [140, 255, 255]),
    "yellow": ([20, 100, 100], [30, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255]),
    "white": ([0, 0, 200], [180, 25, 255]),  # High Value, Low Saturation
    "black": ([0, 0, 0], [180, 255, 50]),  # Low Value
}


# --- Helper Function for Color Detection (Used by OpenCV approach) ---
def get_color_name(hsv_color):
    """
    Identifies the dominant color in an HSV image based on predefined ranges.
    Args:
        hsv_color (numpy.ndarray): The HSV image or ROI.
    Returns:
        str: The name of the detected color, or "Unknown".
    """
    max_pixels = 0
    detected_color = "Unknown"

    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)

        if color_name == "red":  # Handle red's wrap-around
            mask1 = cv2.inRange(hsv_color, lower_bound, upper_bound)
            mask2 = cv2.inRange(
                hsv_color,
                np.array(COLOR_RANGES["red_upper"][0]),
                np.array(COLOR_RANGES["red_upper"][1]),
            )
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name == "red_upper":  # Skip as it's handled by "red"
            continue
        else:
            mask = cv2.inRange(hsv_color, lower_bound, upper_bound)

        # Count non-zero pixels in the mask (i.e., pixels within the color range)
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > max_pixels:
            max_pixels = pixel_count
            detected_color = color_name

    # You might want a threshold here to avoid detecting "Unknown" if very few pixels match
    if max_pixels < (
        hsv_color.shape[0] * hsv_color.shape[1] * 0.05
    ):  # e.g., 5% of pixels
        return "Unknown"

    return detected_color


# --- OpenCV-based Cube Detection Function ---
def detect_cube_opencv(frame):
    """
    Detects cube-like shapes (quadrilaterals) in a given frame and their colors
    using OpenCV's traditional image processing techniques.
    Args:
        frame (numpy.ndarray): The input image frame from the camera.
    Returns:
        tuple: A list of detected cubes (position, color) and the annotated frame.
    """
    detected_cubes = []
    output_frame = frame.copy()

    # 1. Preprocessing
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # 2. Find Contours
    # Find contours in the edged image. RETR_EXTERNAL retrieves only the extreme outer contours.
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 3. Process Contours
    for contour in contours:
        # Approximate the contour with a polygon.
        # epsilon is the maximum distance between the original contour and its approximation.
        # True indicates that the contour is closed.
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.04 * perimeter, True
        )  # 0.04 is a common factor

        # Check if the approximated contour has 4 vertices (a quadrilateral)
        # and if its area is significant enough to be a cube face (avoid small noise)
        area = cv2.contourArea(approx)
        if (
            len(approx) == 4 and area > 1000
        ):  # Minimum area threshold (adjust as needed)
            # Draw the contour on the output frame
            # This line draws the green outline around the detected cube face
            cv2.drawContours(output_frame, [approx], -1, (0, 255, 0), 2)

            # Calculate the centroid of the contour for position
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                position = (cX, cY)

                # Get the bounding rectangle for the ROI
                x, y, w, h = cv2.boundingRect(approx)
                # Ensure the ROI is within frame boundaries
                roi = frame[
                    max(0, y) : min(frame.shape[0], y + h),
                    max(0, x) : min(frame.shape[1], x + w),
                ]

                if roi.size > 0:  # Check if ROI is not empty
                    # Convert ROI to HSV color space for better color detection
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    color_name = get_color_name(hsv_roi)
                else:
                    color_name = "Unknown"

                # Draw position and color text
                cv2.circle(output_frame, position, 5, (0, 0, 255), -1)
                cv2.putText(
                    output_frame,
                    f"Pos: ({cX},{cY})",
                    (cX + 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    output_frame,
                    f"Color: {color_name}",
                    (cX + 10, cY + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                detected_cubes.append({"position": position, "color": color_name})

    return detected_cubes, output_frame


class CubePredictor:
    """
    A class to handle live cube detection using a trained PyTorch Mask R-CNN model.
    It integrates model loading, inference, and visualization, including
    transformation of pixel coordinates to robot coordinates using a homography matrix.
    """

    def __init__(self, model_path, dataset_root, confidence_threshold=0.6):
        """
        Initializes the CubePredictor.

        Args:
            model_path (str): Path to the trained PyTorch model (.pth file).
            dataset_root (str): Root directory of the dataset (e.g., './dataset')
                                to load the test dataset for label mapping.
            confidence_threshold (float): Minimum confidence score for a detection to be displayed.
        """
        self.model_path = model_path
        self.dataset_root = dataset_root
        self.confidence_threshold = confidence_threshold
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"Using device for inference: {self.device}")  # Use logger.info

        self.inference_model = None
        self.label_map = None
        self.homography_matrix = None

        self._load_model()
        self._load_label_map()
        self._load_homography_matrix()

    def _load_model(self):
        """Loads the trained PyTorch model."""
        try:
            # Attempt to load a small part of the dataset to get num_classes
            temp_dataset = CubeDataset(
                root=os.path.join(self.dataset_root, "test"), transforms=None
            )
            num_classes = temp_dataset.num_classes + 1
            del temp_dataset  # Clean up temporary dataset

            self.inference_model = get_model(num_classes)
            self.inference_model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.inference_model.to(self.device)
            self.inference_model.eval()
            logger.info(
                f"PyTorch model loaded from {self.model_path}"
            )  # Use logger.info
        except Exception as e:
            logger.error(
                f"Error loading PyTorch model from {self.model_path}: {e}"
            )  # Use logger.error
            self.inference_model = None  # Ensure model is None if loading fails

    def _load_label_map(self):
        """Loads the label mapping from the dataset."""
        try:
            # Load the test dataset to get the label mapping
            dataset_test = CubeDataset(
                root=os.path.join(self.dataset_root, "test"),
                transforms=get_transform(train=False),
            )
            self.label_map = dataset_test.label_to_cat_name
            logger.info("Label map loaded.")  # Use logger.info
        except Exception as e:
            logger.error(
                f"Error loading label map from dataset: {e}"
            )  # Use logger.error
            self.label_map = {}  # Empty map if loading fails

    def _load_homography_matrix(self):
        """Loads the homography matrix for coordinate transformation."""
        try:
            self.homography_matrix = np.load("homography.npy")
            logger.info(
                "Homography matrix loaded from homography.npy"
            )  # Use logger.info
        except FileNotFoundError:
            logger.warning(
                "homography.npy not found. Please run calibration (e.g., via inference/utils.py) first."
            )  # Use logger.warning
            logger.info("Attempting to run calibration now...")  # Use logger.info
            self.homography_matrix = calibrate_and_transform()
            if self.homography_matrix is not None:
                np.save("homography", self.homography_matrix)
                logger.info(
                    "Homography matrix saved to homography.npy"
                )  # Use logger.info
            else:
                logger.error(
                    "Calibration failed. Robot coordinates will not be available."
                )  # Use logger.error
        except Exception as e:
            logger.error(f"Error loading homography matrix: {e}")  # Use logger.error
            self.homography_matrix = None

    def predict_live(
        self, camera_index: int = 2
    ):  # Add camera_index parameter with default
        """
        Runs live cube detection using the loaded PyTorch model.
        Displays the camera feed with detections and robot coordinates.
        Allows saving frames for training data.

        Args:
            camera_index (int): The index of the webcam to use.
        """
        cap = cv2.VideoCapture(camera_index)  # Use the passed camera_index

        if not cap.isOpened():
            logger.error(
                f"Could not open video stream from camera index {camera_index}."
            )  # Use logger.error
            return

        if self.inference_model is None:
            logger.warning(
                "Model not loaded. Cannot perform PyTorch inference. Falling back to OpenCV detection."
            )  # Use logger.warning
            # Fallback to OpenCV detection if PyTorch model isn't available
            self._run_opencv_live_detection(cap)
            return

        logger.info("\n--- Starting Live Object Detection ---")  # Use logger.info
        logger.info("Press 'q' to quit.")  # Use logger.info
        logger.info(
            f"Press 's' to save the current frame to '{DATA_CAPTURE_DIR}' for training."
        )  # Use logger.info
        logger.info(
            f"Using PyTorch model for detection (confidence threshold: {self.confidence_threshold})."
        )  # Use logger.info

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning(
                    "Could not read frame. Exiting live prediction."
                )  # Use logger.warning
                break

            # Convert frame to RGB and then to PyTorch tensor
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = (
                F.to_tensor(image_rgb).unsqueeze(0).to(self.device)
            )  # Add batch dimension

            with torch.no_grad():
                prediction = self.inference_model(img_tensor)
                logger.debug(
                    f"Raw prediction output: {prediction}"
                )  # Use logger.debug for verbose output

            # Process the output for visualization
            for box, label, score in zip(
                prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"]
            ):
                if score > self.confidence_threshold:  # Confidence threshold
                    box = box.cpu().numpy().astype(int)
                    label_name = self.label_map.get(label.item(), "Unknown")

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
                    )
                    middle = calculate_middle(box[0], box[1], box[2], box[3])
                    cv2.circle(
                        frame, middle, 10, (255, 255, 255), -1
                    )  # Filled white circle

                    robot_coordinates_text = ""
                    if self.homography_matrix is not None:
                        middle_np = np.array(
                            [[[middle[0], middle[1]]]], dtype="float32"
                        )
                        robot_coordinates = cv2.perspectiveTransform(
                            middle_np, self.homography_matrix
                        )[0][0]
                        robot_coordinates_text = f"({robot_coordinates[0]:.2f}, {robot_coordinates[1]:.2f} mm)"
                        logger.debug(
                            f"Transformed pixel {middle} to robot {robot_coordinates_text}"
                        )  # Log transformation

                    # Put label, score, and robot coordinates text
                    label_text = f"{label_name}: {score:.2f} {robot_coordinates_text}"
                    cv2.putText(
                        frame,
                        label_text,
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    logger.info(
                        f"Detected cube: {label_name} at {middle} with score {score:.2f}"
                    )  # Log detection

            cv2.imshow("Live Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Save frame if 's' is pressed
            if key == ord("s"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(DATA_CAPTURE_DIR, f"frame_{timestamp}.png")
                cv2.imwrite(filename, frame)
                logger.info(f"Saved frame to: {filename}")  # Use logger.info
                frame_count += 1
                # Display a temporary message on the frame that it was saved
                temp_frame = frame.copy()
                cv2.putText(
                    temp_frame,
                    f"Saved {frame_count} frames",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Live Object Detection", temp_frame)
                cv2.waitKey(500)  # Show message for 0.5 seconds

            # Break the loop if 'q' is pressed
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam feed closed.")  # Use logger.info

    def _run_opencv_live_detection(self, cap):
        """
        Runs live cube detection using the OpenCV-based approach as a fallback.
        """
        logger.info(
            "\n--- Starting Live OpenCV-based Detection (Fallback) ---"
        )  # Use logger.info
        logger.info("Press 'q' to quit.")  # Use logger.info
        logger.info(
            f"Press 's' to save the current frame to '{DATA_CAPTURE_DIR}' for training."
        )  # Use logger.info

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning(
                    "Could not read frame. Exiting OpenCV fallback detection."
                )  # Use logger.warning
                break

            cubes, annotated_frame = detect_cube_opencv(frame)
            if cubes:
                logger.info(
                    f"Detected cubes (OpenCV fallback): {cubes}"
                )  # Log detected cubes

            cv2.imshow("Cube Detector (OpenCV Fallback)", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            # Save frame if 's' is pressed
            if key == ord("s"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(DATA_CAPTURE_DIR, f"frame_{timestamp}.png")
                cv2.imwrite(filename, frame)
                logger.info(f"Saved frame to: {filename}")  # Use logger.info
                frame_count += 1
                temp_frame = annotated_frame.copy()
                cv2.putText(
                    temp_frame,
                    f"Saved {frame_count} frames",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Cube Detector (OpenCV Fallback)", temp_frame)
                cv2.waitKey(500)

            # Break the loop if 'q' is pressed
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam feed closed (OpenCV fallback).")  # Use logger.info


# if __name__ == "__main__":
#     # Import logging setup from utils
#     from DobotCubeDetector.utils.logging_config import setup_logging
#
#     # Configure logging for standalone execution of this script
#     setup_logging(log_level=logging.INFO)  # Set default log level to INFO
#
#     # Example usage:
#     # Ensure you have a trained model at this path, e.g., after running trainer.py
#     MODEL_PATH = "cube_detector_model.pth"
#     DATASET_ROOT = (
#         "./Pytorch/dataset"  # Original dataset root for label map and num_classes
#     )
#
#     # Instantiate the predictor
#     predictor = CubePredictor(
#         model_path=MODEL_PATH, dataset_root=DATASET_ROOT, confidence_threshold=0.7
#     )
#
#     # Run live prediction
#     predictor.predict_live(camera_index=2)  # Pass camera index here
#
# To run calibration separately (e.g., if homography.npy is missing or needs update)
# homography_matrix = calibrate_and_transform()
# if homography_matrix is not None:
#     np.save("homography", homography_matrix)
#     logger.info("Homography matrix saved.") # Use logger.info
