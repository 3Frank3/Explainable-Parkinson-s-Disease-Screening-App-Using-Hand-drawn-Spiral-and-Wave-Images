from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Small CNN baseline for learning the end-to-end image workflow."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_mobilenetv2(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def create_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    normalized = model_name.lower().replace("-", "_")
    if normalized in {"baseline", "baseline_cnn", "simple_cnn", "cnn"}:
        return BaselineCNN(num_classes=num_classes)
    if normalized in {"mobilenet", "mobilenetv2", "mobile_net_v2"}:
        return build_mobilenetv2(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model: {model_name}")


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    normalized = model_name.lower().replace("-", "_")
    if normalized in {"baseline", "baseline_cnn", "simple_cnn", "cnn"}:
        return model.features[-2]
    if normalized in {"mobilenet", "mobilenetv2", "mobile_net_v2"}:
        return model.features[-1]
    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_name: str,
    image_size: int,
    drawing_type: str,
    metrics: dict | None = None,
) -> None:
    checkpoint = {
        "model_name": model_name,
        "image_size": image_size,
        "drawing_type": drawing_type,
        "num_classes": 2,
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    torch.save(checkpoint, Path(path))


def load_model_from_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    try:
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=map_location)
    model_name = checkpoint.get("model_name", "mobilenetv2")
    model = create_model(model_name, num_classes=checkpoint.get("num_classes", 2), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
