import torch.nn as nn
import torchvision.models as models

def build_model():
    """
    Builds a simple baseline CNN model using ResNet18.
    """

    # Load pretrained resnet18
    model = models.resnet18(weights="DEFAULT")

    # Replace final layer for binary classification (cat vs dog)
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, 2)

    return model


if __name__ == "__main__":
    model = build_model()
    print("Model created successfully")
