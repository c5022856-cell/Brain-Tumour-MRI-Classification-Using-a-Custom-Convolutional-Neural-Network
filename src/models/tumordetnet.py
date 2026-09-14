from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class TumorDetNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, base_channels: int = 32, dropout: float = 0.4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            nn.MaxPool2d(2),
            ConvBlock(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 4, base_channels * 8),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        return self.classifier(encoded)

