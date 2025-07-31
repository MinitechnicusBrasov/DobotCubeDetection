import logging
import os


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Sets up the standardized logging configuration for the application.

    Args:
        log_level (int): The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
        log_file (str, optional): Path to a file where logs should also be written.
                                  If None, logs only go to the console.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers to prevent duplicate output if called multiple times
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Define the formatter for all handlers
    # %(asctime)s: Timestamp
    # %(name)s: Name of the logger (e.g., DobotCubeDetector.utils.data_capture)
    # %(levelname)s: Log level (e.g., INFO, ERROR)
    # %(message)s: The actual log message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler: Always output to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (optional): Output to a file if log_file is provided
    if log_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Inform about logging setup (this will go to console)
    root_logger.info(f"Logging configured at level: {logging.getLevelName(log_level)}")
    if log_file:
        root_logger.info(f"Logs also being written to file: {log_file}")
