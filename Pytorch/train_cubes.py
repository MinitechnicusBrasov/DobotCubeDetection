import os
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

# --- 3. Training and Evaluation ---
def train_one_epoch(model, optimizer, data_loader, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # The model returns a dictionary of losses in training mode
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        total_loss += loss_value

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch training loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # In evaluation, we also pass targets to get validation loss
        loss_dict, detections = eval_forward(model, images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        total_loss += loss_value

    avg_loss = total_loss / len(data_loader)
    print(f"Validation loss: {avg_loss:.4f}")
    return avg_loss

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    DATASET_ROOT = "./dataset" # IMPORTANT: Update this path
    TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
    VALID_DIR = os.path.join(DATASET_ROOT, 'valid')
    TEST_DIR = os.path.join(DATASET_ROOT, 'test')
    MODEL_SAVE_PATH = "cube_detector_model.pth"
    
    NUM_EPOCHS = 70
    BATCH_SIZE = 5
    LEARNING_RATE = 0.005

    # --- Setup ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    #
    # # Initialize datasets
    dataset_train = CubeDataset(root=TRAIN_DIR, transforms=get_transform(train=True))
    dataset_valid = CubeDataset(root=VALID_DIR, transforms=get_transform(train=False))

    # # The number of classes is the number of cube colors + 1 for the background
    num_classes = dataset_train.num_classes + 1

    # # Initialize data loaders
    data_loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn)

    data_loader_valid = DataLoader(
        dataset_valid, batch_size=1, shuffle=False,
        collate_fn=collate_fn)

    # # --- Model Training ---
    model = get_model(num_classes).to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_one_epoch(model, optimizer, data_loader_train, device)
        evaluate(model, data_loader_valid, device)

    print("\n--- Training Finished ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


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
    if len(dataset_test) == 0:
        print("Test dataset is empty. Skipping inference visualization.")
    else:
        img, _ = dataset_test[1] # Take the first test image
        label_map = dataset_test.label_to_cat_name

        with torch.no_grad():
            prediction = inference_model([img.to(device)])
            print(prediction)

        # Process the output for visualization
        image_np = img.mul(255).permute(1, 2, 0).byte().numpy()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            if score > 0.8: # Confidence threshold
                print("Confidence reached")
                box = box.cpu().numpy().astype(int)
                label_name = label_map.get(label.item(), 'Unknown')
                
                # Draw bounding box (the position)
                cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Put label and score text (the color)
                label_text = f"{label_name}: {score:.2f}"
                cv2.putText(image_np, label_text, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save or display the result
        cv2.imwrite("test_output.jpg", image_np)
        print("Inference result saved to test_output.jpg")
        # To display in a window (if you have a GUI environment):
        cv2.imshow("Inference Result", image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
