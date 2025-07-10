import cv2
import numpy as np
import os
import datetime

# --- TensorFlow Placeholder Imports ---
# These imports are for demonstrating where TensorFlow would be used.
# You will need to install TensorFlow: pip install tensorflow
# For actual use, you'd load a pre-trained model and fine-tune it.
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuration ---
# Directory to save captured images for dataset creation
DATA_CAPTURE_DIR = "captured_images_for_training"
if not os.path.exists(DATA_CAPTURE_DIR):
    os.makedirs(DATA_CAPTURE_DIR)

# Define color ranges in HSV. These values might need adjustment based on your lighting
# and the specific colors of your cubes.
# Format: (Hue_min, Saturation_min, Value_min), (Hue_max, Saturation_max, Value_max)
COLOR_RANGES = {
    "red": ([0, 100, 100], [10, 255, 255]),
    "red_upper": ([170, 100, 100], [180, 255, 255]), # Red wraps around in HSV
    "green": ([40, 40, 40], [80, 255, 255]),
    "blue": ([100, 100, 100], [140, 255, 255]),
    "yellow": ([20, 100, 100], [30, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255]),
    "white": ([0, 0, 200], [180, 25, 255]), # High Value, Low Saturation
    "black": ([0, 0, 0], [180, 255, 50]),   # Low Value
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

        if color_name == "red": # Handle red's wrap-around
            mask1 = cv2.inRange(hsv_color, lower_bound, upper_bound)
            mask2 = cv2.inRange(hsv_color, np.array(COLOR_RANGES["red_upper"][0]), np.array(COLOR_RANGES["red_upper"][1]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name == "red_upper": # Skip as it's handled by "red"
            continue
        else:
            mask = cv2.inRange(hsv_color, lower_bound, upper_bound)

        # Count non-zero pixels in the mask (i.e., pixels within the color range)
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > max_pixels:
            max_pixels = pixel_count
            detected_color = color_name

    # You might want a threshold here to avoid detecting "Unknown" if very few pixels match
    if max_pixels < (hsv_color.shape[0] * hsv_color.shape[1] * 0.05): # e.g., 5% of pixels
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
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Process Contours
    for contour in contours:
        # Approximate the contour with a polygon.
        # epsilon is the maximum distance between the original contour and its approximation.
        # True indicates that the contour is closed.
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) # 0.04 is a common factor

        # Check if the approximated contour has 4 vertices (a quadrilateral)
        # and if its area is significant enough to be a cube face (avoid small noise)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 1000: # Minimum area threshold (adjust as needed)
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
                roi = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]

                if roi.size > 0: # Check if ROI is not empty
                    # Convert ROI to HSV color space for better color detection
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    color_name = get_color_name(hsv_roi)
                else:
                    color_name = "Unknown"

                # Draw position and color text
                cv2.circle(output_frame, position, 5, (0, 0, 255), -1)
                cv2.putText(output_frame, f"Pos: ({cX},{cY})", (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output_frame, f"Color: {color_name}", (cX + 10, cY + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detected_cubes.append({"position": position, "color": color_name})

    return detected_cubes, output_frame

# --- Conceptual TensorFlow Training Function ---
def train_tensorflow_model(dataset_path=DATA_CAPTURE_DIR, model_save_path="trained_cube_detector_model.h5"):
    """
    Conceptual function to outline the steps for training a TensorFlow model.
    This function is NOT runnable as-is and requires a properly annotated dataset
    and a full TensorFlow training pipeline setup.

    Args:
        dataset_path (str): Path to the directory containing images and annotations.
        model_save_path (str): Path to save the trained model.
    """
    print("\n--- Initiating Conceptual TensorFlow Training ---")
    print(f"Dataset path: {dataset_path}")
    print("1. Load and preprocess your annotated images and bounding box/color labels.")
    print("   This involves parsing annotation files (e.g., XML, JSON) for each image.")
    print("2. Define your TensorFlow model architecture (e.g., MobileNetV2, EfficientDet).")
    print("   For object detection, you'd typically use a pre-trained model from TensorFlow Model Garden.")
    print("3. Compile the model with an optimizer, loss function, and metrics.")
    print("4. Train the model on your dataset (e.g., model.fit(...)).")
    print("   This will involve many epochs and can take a long time depending on your data and hardware.")
    print("5. Evaluate the model's performance on a validation set.")
    print(f"6. Save the trained model to: {model_save_path}")
    print("--- Conceptual TensorFlow Training Complete ---")
    # Example placeholder for saving a dummy model
    # dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    # dummy_model.save(model_save_path)
    # print(f"Dummy model saved to {model_save_path}")

# --- Conceptual TensorFlow Prediction Function ---
# Global variable to hold the loaded TensorFlow model (if enabled)
# tf_model = None
# tf_class_names = ["red", "green", "blue", "yellow", "orange", "white", "black", "unknown"] # Example classes

def predict_with_tensorflow_model(frame, model_path="trained_cube_detector_model.h5"):
    """
    Conceptual function to make predictions using a trained TensorFlow model.
    This function assumes a model has been trained and saved.
    It will return dummy data if the model is not loaded or available.

    Args:
        frame (numpy.ndarray): The input image frame from the camera.
        model_path (str): Path to the trained TensorFlow model file.
    Returns:
        tuple: A list of detected cubes (position, color) and the annotated frame.
    """
    detected_cubes = []
    output_frame = frame.copy()

    # --- Conceptual Model Loading (uncomment and implement for real use) ---
    # global tf_model, tf_class_names
    # if tf_model is None:
    #     try:
    #         tf_model = load_model(model_path)
    #         print(f"TensorFlow model loaded from {model_path}")
    #     except Exception as e:
    #         print(f"Error loading TensorFlow model: {e}. Using dummy predictions.")
    #         # Fallback to dummy data if model loading fails
    #         # This is where you might revert to OpenCV if TF fails
    #         return detect_cube_opencv(frame)

    # --- Conceptual Preprocessing for TensorFlow Model ---
    # if tf_model:
    #     # Resize and preprocess the frame according to your model's input requirements
    #     input_image_size = (224, 224) # Example size, adjust to your model's input
    #     input_frame = cv2.resize(frame, input_image_size)
    #     input_frame = img_to_array(input_frame)
    #     input_frame = np.expand_dims(input_frame, axis=0) # Add batch dimension
    #     input_frame = preprocess_input(input_frame) # Specific to your model's preprocessing

    #     # --- Conceptual Prediction ---
    #     # predictions = tf_model.predict(input_frame)
    #     # For object detection, predictions would typically include bounding boxes,
    #     # class probabilities, and confidence scores.

    #     # --- Conceptual Parsing of Predictions ---
    #     # Iterate through predictions and draw results
    #     # For demonstration, let's add a dummy cube prediction
    #     dummy_bbox = [50, 50, 200, 200] # [x_min, y_min, x_max, y_max]
    #     dummy_color_id = 0 # Corresponds to "red" in tf_class_names
    #     dummy_confidence = 0.95

    #     if dummy_confidence > 0.5: # Example confidence threshold
    #         x1, y1, x2, y2 = dummy_bbox
    #         color_name = tf_class_names[dummy_color_id]
    #         cX = (x1 + x2) // 2
    #         cY = (y1 + y2) // 2
    #         position = (cX, cY)

    #         cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for TF detection
    #         cv2.putText(output_frame, f"TF Pos: ({cX},{cY})", (cX + 10, cY - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    #         cv2.putText(output_frame, f"TF Color: {color_name}", (cX + 10, cY + 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    #         detected_cubes.append({"position": position, "color": color_name, "source": "TensorFlow"})
    # else:
    #     print("TensorFlow model not loaded. Falling back to OpenCV detection.")
    #     return detect_cube_opencv(frame)

    # For demonstration, if TF is not truly integrated, we'll just use OpenCV
    # In a real scenario, you'd uncomment the TF logic above and remove this line
    return detect_cube_opencv(frame)


def main():
    """
    Main function to run the cube detection application.
    """
    # --- Option to conceptually train the model (uncomment to 'run' this step) ---
    # train_tensorflow_model()

    # Open the default camera (usually 0)
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Press 'q' to quit.")
    print(f"Press 's' to save the current frame to '{DATA_CAPTURE_DIR}' for training.")
    print("Currently using OpenCV for detection. For TensorFlow, you'd integrate a trained model.")

    frame_count = 0
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # --- Choose your detection method ---
        # For real TensorFlow prediction, you would uncomment the line below
        # and comment out the detect_cube_opencv line.
        # cubes, annotated_frame = predict_with_tensorflow_model(frame)
        cubes, annotated_frame = detect_cube_opencv(frame)


        # Display the annotated frame
        cv2.imshow("Cube Detector", annotated_frame)

        # Print detected cubes information to console
        if cubes:
            # print(f"Detected Cubes: {cubes}") # Commented out to reduce console spam
            pass

        key = cv2.waitKey(1) & 0xFF

        # Save frame if 's' is pressed
        if key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(DATA_CAPTURE_DIR, f"frame_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved frame to: {filename}")
            frame_count += 1
            cv2.putText(annotated_frame, f"Saved {frame_count} frames", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Cube Detector", annotated_frame) # Update display immediately after saving

        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break

    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

