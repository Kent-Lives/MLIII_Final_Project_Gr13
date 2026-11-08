from typing import cast
from torchvision import models
import torch.nn as nn

### resnet18 - pre trained model
def get_pretrained_model(
    model_name="resnet18",
    num_classes=2,
    feature_extract=True
):

    if model_name == "resnet18":

        # Load pretrained ResNet18
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze pretrained weights
        if feature_extract:
            for parameter in model.parameters():
                parameter.requires_grad = False

        # Replace final layer for binary classification
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    ### mobilenetv2 - pre trained model
    elif model_name == "mobilenet_v2":

        # Load pretrained MobileNetV2
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # Freeze pretrained weights
        if feature_extract:
            for parameter in model.parameters():
                parameter.requires_grad = False

        # Replace final layer for binary classification.
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    else:
        raise ValueError(
            f"Model {model_name} is unsupported."
        )

    return model
