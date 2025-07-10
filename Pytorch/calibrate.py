import numpy as np
import cv2
from lib import calculate_middle

BOX_1 = [20, 20, 120, 120]
BOX_2 = [500, 20, 600, 120]
BOX_3 = [20, 300, 120, 400]
BOX_4 = [500, 300, 600, 400]

MIDDLE_1 = calculate_middle(*BOX_1)
MIDDLE_2 = calculate_middle(*BOX_2)
MIDDLE_3 = calculate_middle(*BOX_3)
MIDDLE_4 = calculate_middle(*BOX_4)


def show_calibration():
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            return

        cv2.rectangle(frame, (BOX_1[0], BOX_1[1]), (BOX_1[2], BOX_1[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Position 1", (BOX_1[0], BOX_1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, MIDDLE_1, radius=10, color=(255, 255, 255), thickness=1)

        cv2.rectangle(frame, (BOX_2[0], BOX_2[1]), (BOX_2[2], BOX_2[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Position 2", (BOX_2[0], BOX_2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, MIDDLE_2, radius=10, color=(255, 255, 255), thickness=1)

        cv2.rectangle(frame, (BOX_3[0], BOX_3[1]), (BOX_3[2], BOX_3[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Position 3", (BOX_3[0], BOX_3[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, MIDDLE_3, radius=10, color=(255, 255, 255), thickness=1)

        cv2.rectangle(frame, (BOX_4[0], BOX_4[1]), (BOX_4[2], BOX_4[3]), (0, 255, 0), 2)
        cv2.putText(frame, "Position 4", (BOX_4[0], BOX_4[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, MIDDLE_4, radius=10, color=(255, 255, 255), thickness=1)

        
        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return


def calibrate_and_transform():
    """
    Calculates the homography matrix and transforms a point.
    """
    # --- 1. DEFINE YOUR CALIBRATION POINTS ---
    # IMPORTANT: Replace these with your own measured data.
    # The order of points must be the same in both arrays.
    show_calibration()
        
    # (X, Y) coordinates from your robot's workspace in millimeters
    robot_points = np.array([
        [353.1, 89.4],      # Point 1 (e.g., Top-Left)
        [355.8, -41.5],    # Point 2 (e.g., Top-Right)
        [281.6, -43.9],  # Point 3 (e.g., Bottom-Right)
        [277.0, 88.7]     # Point 4 (e.g., Bottom-Left)
    ], dtype="float32")

    # (x, y) coordinates of the same points from your camera image in pixels
    pixel_points = np.array([
        [MIDDLE_1[0], MIDDLE_1[1]],      # Point 1 (pixel coords of robot's 0,0)
        [MIDDLE_2[0], MIDDLE_2[1]],      # Point 2 (pixel coords of robot's 300,0)
        [MIDDLE_4[0], MIDDLE_4[1]],      # Point 3 (pixel coords of robot's 300,200)
        [MIDDLE_3[0], MIDDLE_3[1]]       # Point 4 (pixel coords of robot's 0,200)
    ], dtype="float32")

    # --- 2. CALCULATE THE HOMOGRAPHY MATRIX ---
    # This matrix 'h' is your calibration key. You can save it and reuse it.
    h, status = cv2.findHomography(pixel_points, robot_points)
    
    if h is None:
        print("Homography calculation failed. Check your points.")
        return

    print("✅ Homography Matrix Calculated. You can save and reuse this matrix:\n", h)
    
    # --- 3. TRANSFORM A NEW POINT ---
    # Let's say your object detection script found a cube centered at (350, 250) pixels.
    pixel_center_of_cube = np.array([[[350, 250]]], dtype="float32")

    # Use the matrix to transform the pixel point to a robot coordinate
    robot_coord = cv2.perspectiveTransform(pixel_center_of_cube, h)

    # The result is a nested array, so we extract the values.
    rx, ry = robot_coord[0][0]

    print("\n--- Transformation Result ---")
    print(f"Pixel Center: (px={pixel_center_of_cube[0][0][0]}, py={pixel_center_of_cube[0][0][1]})")
    print(f"Robot Coordinate: (RX={rx:.2f} mm, RY={ry:.2f} mm)")
    
    return h

if __name__ == '__main__':
    # Run the calibration and get the matrix
    homography_matrix = calibrate_and_transform()
    np.save("homography", homography_matrix)
    
    # You would save this `homography_matrix` and load it in your main robot control script.
