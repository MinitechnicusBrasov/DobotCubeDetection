# DobotCubeDetector

## 📝 Project Overview

The DobotCubeDetector project is an advanced robotic vision system designed to enable a Dobot robotic arm to detect and identify a Rubik's Cube. It leverages a deep learning model for computer vision and a custom calibration pipeline to translate pixel coordinates from a camera feed into real-world coordinates for the Dobot Magician.

This system provides a robust solution for tasks like cube manipulation, automated sorting, or other projects requiring high-precision object detection and interaction with a robotic arm.

## ✨ Features

* **Instance Segmentation:** Uses a pre-trained Mask R-CNN model to accurately detect and segment Rubik's Cubes in a live camera feed.

* **Camera-to-Robot Calibration:** Employs homography to calculate a transformation matrix, enabling the seamless conversion of 2D pixel coordinates to 3D robot workspace coordinates.

* **Modular Architecture:** Designed with separate modules for model transformations, inference utilities, and the main application logic, making the code easy to understand and extend.

* **Custom Loss Calculation:** Includes a custom `eval_forward` function to calculate and log losses during validation, which is essential for model training and debugging.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following hardware and software installed:

* **Python 3.8+**

* **Dobot Magician** robotic arm

* **DobotStudio** or a similar Dobot control software

* **USB Camera** compatible with OpenCV

* A machine with **CUDA** support for GPU acceleration (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/DobotCubeDetector.git
cd DobotCubeDetector
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## 🛠️ Usage

This project uses a single command-line interface (`cli.py`) to manage all its functionalities. All commands are executed from the project's root directory.

### Command-Line Interface (CLI)

The `cli.py` script is the main entry point for the application. You can view all available commands and options by using the `--help` flag.

```bash
python cli.py --help
```

---

### 1. Training the Model

To train a new model, use the `train` command. You can specify the dataset location, number of epochs, and other hyperparameters.

**Example:**

```bash
python cli.py train --epochs 50 --batch_size 8 --dataset_root ./data/my_dataset/
```

---

### 2. Live Prediction

After a model is trained, you can use the `predict` command to run live object detection on a webcam feed.

**Example:**

```bash
python cli.py calibrate --homography_save_path homography.npy
```

---

### 4. Data Capture

If you need to create a new dataset for training, use the `capture-data` command. This opens a camera feed, allowing you to capture images for later annotation.

**Example:**

```bash
python cli.py capture-data --output_dir ./data/raw_images --camera_index 1
```

## 📂 Project Structure

```
DobotCubeDetector/
├── DobotCubeDetector/
│   ├── inference/
│   │   ├── init.py
│   │   └── utils.py          # Contains camera calibration and eval_forward logic.
│   ├── models/
│   │   ├── init.py
│   │   └── transforms.py     # Defines image transformations and augmentations.
│   ├── utils/
│   │   └── logging_config.py # Sets up project-wide logging.
│   ├── main.py               # Application logic for the main loop (now called by the CLI).
│   └── ...                   # Other project files.
├── cli.py                    # Command-Line Interface for the entire project.
├── README.md                 # This file.
└── requirements.txt
```

## 🤝 Contributing

We welcome contributions to this project! Feel free to open an issue or submit a pull request for new features, bug fixes, or improvements.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
