from __future__ import annotations

from src.models.baseline_cnn import BaselineCNN
from src.models.tumordetnet import TumorDetNet


def build_model(model_name: str, num_classes: int, channels: int, model_cfg: dict) -> object:
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, in_channels=channels, dropout=float(model_cfg["dropout"]))
    if model_name == "tumordetnet":
        return TumorDetNet(
            num_classes=num_classes,
            in_channels=channels,
            base_channels=int(model_cfg["base_channels"]),
            dropout=float(model_cfg["dropout"]),
        )
    raise ValueError(f"Unsupported model: {model_name}")

