"""Small CNN, ResNet-18, and checkpoint loading."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [32, 64, 128, 192, 256]
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(256, 2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_resnet18(pretrained=False):
    """Download ImageNet weights only when explicitly preparing a training run."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_model(model_name, checkpoint_path, device="cpu"):
    device = torch.device(device)
    if model_name == "resnet18":
        model = build_resnet18()
    elif model_name == "small_cnn":
        model = SmallCNN()
    else:
        raise ValueError("model_name must be resnet18 or small_cnn")
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    threshold = float(checkpoint["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid checkpoint threshold")
    return model.to(device).eval(), threshold
