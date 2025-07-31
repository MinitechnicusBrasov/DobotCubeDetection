import argparse
import os
import sys
import torch
import numpy as np
import logging  # Import logging for CLI-specific messages

# Ensure the parent directory (DobotCubeDetector) is in the Python path
# so that relative imports work when running this script directly.
# This is crucial when running 'python -m DobotCubeDetector.cli' or 'python cli.py'
# from the DobotCubeDetector/ directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import functionalities from your restructured project
from DobotCubeDetector.inference.predictor import CubePredictor
from DobotCubeDetector.training.trainer import train_one_epoch, evaluate
from DobotCubeDetector.training.dataset import CubeDataset, collate_fn
from DobotCubeDetector.models.mask_rcnn_model import get_model
from DobotCubeDetector.models.transforms import get_transform
from DobotCubeDetector.inference.utils import calibrate_and_transform
from DobotCubeDetector.utils.data_capture import (
    run_image_capture_mode,
    DEFAULT_CAPTURE_DIR,
)
from DobotCubeDetector.utils.logging_config import (
    setup_logging,
)  # Import the setup function

cli_logger = logging.getLogger(__name__)


def main():
    """
    Main function for the Dobot Cube Detector CLI.
    Sets up argument parsing for 'train', 'predict', 'calibrate', and 'capture-data' commands.
    """

    ROOT_DIR = os.path.dirname(__file__)
    DEFAULT_DATASET_DIR = os.path.join(ROOT_DIR, "data", "dataset")
    DEFAULT_MODEL_SAVE_MODEL = os.path.join(
        ROOT_DIR, "models", "weights", "detect_cube.pth"
    )

    setup_logging(log_level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Dobot Cube Detector: Train, Predict, Calibrate, or Capture Data.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better formatting of help messages
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train Subparser ---
    train_parser = subparsers.add_parser(
        "train",
        help="Train the cube detection model.",
        description="""
        Trains the PyTorch Mask R-CNN model on your dataset.
        Ensure your dataset is in COCO format with '_annotations.coco.json'
        in the specified train/valid directories.
        """,
    )

    train_parser.add_argument(
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_DIR,  # Adjust this default if your dataset is elsewhere
        help=f"Root directory of your dataset (e.g., '{DEFAULT_DATASET_DIR}'). "
        "Expected structure: dataset_root/train/, dataset_root/valid/, dataset_root/test/.",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=70, help="Number of training epochs."
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for training."
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.005, help="Learning rate for the optimizer."
    )
    train_parser.add_argument(
        "--model_save_path",
        type=str,
        default=DEFAULT_MODEL_SAVE_MODEL,
        help="Path to save the trained model weights.",
    )
    train_parser.set_defaults(func=run_train)

    # --- Predict Subparser ---
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run live cube detection using a trained model.",
        description="""
        Runs live inference using a pre-trained Mask R-CNN model.
        Requires a webcam (camera index 2 by default) and a trained model file.
        """,
    )
    predict_parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_SAVE_MODEL,
        help="Path to the trained model weights (.pth file).",
    )
    predict_parser.add_argument(
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_DIR,  # Used to load label map for prediction
        help=f"Root directory of your dataset (e.g., '{DEFAULT_DATASET_DIR}'). "
        "Used to get class names for visualization.",
    )
    predict_parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for displaying detections (0.0 to 1.0).",
    )
    predict_parser.add_argument(
        "--camera_index",
        type=int,
        default=2,
        help="Index of the webcam to use (e.g., 0 for default, 1, 2, etc.).",
    )
    predict_parser.set_defaults(func=run_predict)

    # --- Calibrate Subparser ---
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Perform camera-to-robot coordinate calibration.",
        description="""
        Calculates and saves the homography matrix for transforming
        pixel coordinates to robot coordinates.
        This will open a calibration window. Follow the instructions.
        """,
    )
    calibrate_parser.add_argument(
        "--homography_save_path",
        type=str,
        default="homography.npy",
        help="Path to save the calculated homography matrix.",
    )
    calibrate_parser.set_defaults(func=run_calibrate)

    # --- Capture Data Subparser ---
    capture_data_parser = subparsers.add_parser(
        "capture-data",
        help="Capture images from camera for dataset labeling.",
        description="""
        Opens a live camera feed and allows you to capture frames
        by pressing 's'. Captured images are saved to a specified directory.
        These images can then be used for manual annotation and dataset creation.
        """,
    )
    capture_data_parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_CAPTURE_DIR,  # Now imported from data_capture.py
        help=f"Directory to save captured images. Default: '{DEFAULT_CAPTURE_DIR}'.",
    )
    capture_data_parser.add_argument(
        "--camera_index",
        type=int,
        default=2,
        help="Index of the webcam to use (e.g., 0 for default, 1, 2, etc.).",
    )
    capture_data_parser.set_defaults(func=run_capture_data)

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()  # If no subcommand is given, print general help


def run_train(args):
    """Executes the training process."""
    cli_logger.info("\n--- Starting Training with parameters: ---")
    cli_logger.info(f"Dataset Root: {args.dataset_root}")
    cli_logger.info(f"Epochs: {args.epochs}")
    cli_logger.info(f"Batch Size: {args.batch_size}")
    cli_logger.info(f"Learning Rate: {args.lr}")
    cli_logger.info(f"Model Save Path: {args.model_save_path}")
    cli_logger.info("------------------------------------------")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cli_logger.info(f"Using device: {device}")

    train_dir = os.path.join(args.dataset_root, "train")
    valid_dir = os.path.join(args.dataset_root, "valid")

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        cli_logger.error("Training or validation dataset directory not found.")
        cli_logger.error(f"Expected: {train_dir} and {valid_dir}")
        sys.exit(1)

    dataset_train = CubeDataset(root=train_dir, transforms=get_transform(train=True))
    dataset_valid = CubeDataset(root=valid_dir, transforms=get_transform(train=False))

    num_classes = dataset_train.num_classes + 1  # +1 for background

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    model = get_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(args.epochs):
        cli_logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, optimizer, data_loader_train, device)
        evaluate(model, data_loader_valid, device)

    torch.save(model.state_dict(), args.model_save_path)
    cli_logger.info(
        f"\n--- Training Finished! Model saved to {args.model_save_path} ---"
    )


def run_predict(args):
    """Executes the live prediction process."""
    cli_logger.info(f"--- Starting Live Prediction with parameters: ---")
    cli_logger.info(f"Model Path: {args.model_path}")
    cli_logger.info(f"Dataset Root (for labels): {args.dataset_root}")
    cli_logger.info(f"Confidence Threshold: {args.confidence}")
    cli_logger.info(f"Camera Index: {args.camera_index}")
    cli_logger.info("-------------------------------------------------")

    if not os.path.exists(args.model_path):
        cli_logger.error(f"Trained model not found at {args.model_path}.")
        cli_logger.error("Please train the model first or provide a valid path.")
        sys.exit(1)

    # Initialize CubePredictor with the provided arguments
    predictor = CubePredictor(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        confidence_threshold=args.confidence,
    )
    # Pass the camera index to the predict_live method
    predictor.predict_live(camera_index=args.camera_index)


def run_calibrate(args):
    """Executes the calibration process."""
    cli_logger.info(f"\n--- Starting Camera-to-Robot Calibration ---")
    cli_logger.info(f"Homography matrix will be saved to: {args.homography_save_path}")
    cli_logger.info("Follow the instructions in the calibration window.")
    cli_logger.info("---------------------------------------------")

    homography_matrix = calibrate_and_transform()
    if homography_matrix is not None:
        np.save(args.homography_save_path, homography_matrix)
        cli_logger.info(
            f"Calibration complete! Homography matrix saved to {args.homography_save_path}"
        )
    else:
        cli_logger.error("Calibration failed. Homography matrix not saved.")


def run_capture_data(args):
    """Executes the image capture for labeling process."""
    run_image_capture_mode(output_dir=args.output_dir, camera_index=args.camera_index)


if __name__ == "__main__":
    main()
