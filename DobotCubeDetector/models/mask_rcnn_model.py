import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model(num_classes):
    """
    Loads a pre-trained Mask R-CNN model and modifies its classifier
    head for the number of classes in the custom dataset.

    Args:
        num_classes (int): The total number of classes in the dataset,
                           including the background class.

    Returns:
        torchvision.models.detection.MaskRCNN: The configured Mask R-CNN model.
    """
    # Load a model pre-trained on COCO
    # Using 'DEFAULT' for weights automatically downloads the recommended weights
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the box classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one that has num_classes outputs
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer_dim = 256 # A common hidden layer dimension for the mask head

    # Replace the pre-trained mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer_dim, num_classes
    )

    return model
