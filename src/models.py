"""ImageNet-pretrained architectures with new 200-class classifier heads."""
from __future__ import annotations

from torch import nn
from torchvision import models

VALID_MODELS = ("resnet18", "efficientnet_b0")


def create_model(name: str, num_classes: int = 200, pretrained: bool = True):
    """Return the model and its classification head for differential learning rates."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, model.fc
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model, model.classifier
    raise ValueError(f"Unknown model {name!r}; choose from {VALID_MODELS}")
