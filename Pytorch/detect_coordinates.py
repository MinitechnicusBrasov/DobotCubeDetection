import os
from dataset import CubeDataset
import json
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from eval_forward import eval_forward
from dataset import CubeDataset, get_model, get_transform, collate_fn
from lib import calculate_middle


def main():
    # --- Configuration ---
    DATASET_ROOT = "./dataset" # IMPORTANT: Update this path
    TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
    VALID_DIR = os.path.join(DATASET_ROOT, 'valid')
    TEST_DIR = os.path.join(DATASET_ROOT, 'test')
    MODEL_SAVE_PATH = "cube_detector_model.pth"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    #
    # # Initialize datasets
    dataset_train = CubeDataset(root=TRAIN_DIR, transforms=get_transform(train=True))
    dataset_valid = CubeDataset(root=VALID_DIR, transforms=get_transform(train=False))

    # # The number of classes is the number of cube colors + 1 for the background
    num_classes = dataset_train.num_classes + 1

    # # Initialize data loaders

    # # --- Model Training ---
    # model = get_model(num_classes).to(device)
    #
    # # Optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    #
    # print("\n--- Starting Training ---")
    # for epoch in range(NUM_EPOCHS):
    #     print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    #     train_one_epoch(model, optimizer, data_loader_train, device)
    #     evaluate(model, data_loader_valid, device)
    #
    # print("\n--- Training Finished ---")
    # torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # print(f"Model saved to {MODEL_SAVE_PATH}")


    # --- 5. Inference and Visualization ---
    print("\n--- Running Inference on a Test Image ---")
    
    # Load the trained model
    inference_model = get_model(num_classes)
    inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    inference_model.to(device)
    inference_model.eval()

    # Load a test image and its annotations
    dataset_test = CubeDataset(root=TEST_DIR, transforms=get_transform(train=False))
    
    # Check if test set is not empty
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    homography_matrix = np.load("homography.npy")
    print(f"Matrix: {homography_matrix}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(image_rgb) # Take the first test image
        img = img.unsqueeze(0).to(device)
        label_map = dataset_test.label_to_cat_name

        with torch.no_grad():
            prediction = inference_model(img)

        # Process the output for visualization
        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            if score > 0.6: # Confidence threshold
                box = box.cpu().numpy().astype(int)
                label_name = label_map.get(label.item(), 'Unknown')
                
            # Draw bounding box (the position)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                middle = calculate_middle(box[0], box[1], box[2], box[3])
                cv2.circle(frame, middle, 10, (255, 255, 255), 1)
                middle_np = np.array([[[middle[0], middle[1]]]], dtype="float32")
                robot_coordinates = cv2.perspectiveTransform(middle_np, homography_matrix)[0][0]
                
            # Put label and score text (the color)
                label_text = f"{label_name}: {score:.2f}: ({robot_coordinates[0]:.2f}, {robot_coordinates[1]:.2f})"
                cv2.putText(frame, label_text, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Live  Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed closed.")

if __name__ == "__main__":
    main()
