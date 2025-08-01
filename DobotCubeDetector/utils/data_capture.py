import os
import cv2
import datetime
import logging  # Import the logging module

# Get the logger for this module. It will inherit configuration from the root logger.
logger = logging.getLogger(__name__)

# Get the path to the DobotCubeDetector package root
# This assumes data_capture.py is inside DobotCubeDetector/utils/
# So, go up one level from 'utils' to 'DobotCubeDetector'
_dobot_cube_detector_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Define a default directory for captured images for labeling
# This will now always be located inside the DobotCubeDetector package.
DEFAULT_CAPTURE_DIR = os.path.join(
    _dobot_cube_detector_root, "data", "raw_image_captured"
)

# Ensure the directory exists when this module is imported
if not os.path.exists(DEFAULT_CAPTURE_DIR):
    os.makedirs(DEFAULT_CAPTURE_DIR)
    logger.info(
        f"Created default capture directory: {DEFAULT_CAPTURE_DIR}"
    )  # Use logger.info


def run_image_capture_mode(output_dir: str, camera_index: int):
    """
    Runs a live camera feed, allowing the user to capture images by pressing 's'.
    Captured images are saved to the specified output directory.

    Args:
        output_dir (str): The directory where captured images will be saved.
        camera_index (int): The index of the webcam to use.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")  # Use logger.info

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error(
            f"Could not open video stream from camera index {camera_index}."
        )  # Use logger.error
        logger.error(
            "Please check if the camera is connected and the index is correct."
        )  # Use logger.error
        return

    logger.info(f"\n--- Starting Image Capture Mode ---")  # Use logger.info
    logger.info(f"Images will be saved to: {output_dir}")  # Use logger.info
    logger.info("Press 's' to save the current frame.")  # Use logger.info
    logger.info("Press 'q' to quit.")  # Use logger.info
    logger.info("-----------------------------------")  # Use logger.info

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            logger.warning(
                "Could not read frame. Exiting capture mode."
            )  # Use logger.warning
            break

        # Display the current frame
        cv2.imshow("Image Capture for Labeling (Press 's' to Save, 'q' to Quit)", frame)

        key = cv2.waitKey(1) & 0xFF

        # Save frame if 's' is pressed
        if key == ord("s"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(output_dir, f"frame_{timestamp}.png")
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
            cv2.imshow(
                "Image Capture for Labeling (Press 's' to Save, 'q' to Quit)",
                temp_frame,
            )
            cv2.waitKey(500)  # Show message for 0.5 seconds

        # Break the loop if 'q' is pressed
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Image capture mode closed.")  # Use logger.info
