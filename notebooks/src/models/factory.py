import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes, channels=3, dropout=0.3, base_channels=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, base_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(base_channels * 4, num_classes))
    def forward(self, x): return self.classifier(self.features(x))

def build_model(model_name, num_classes, channels, model_cfg):
    if model_name not in {'baseline_cnn', 'tumordetnet'}: raise ValueError(f'Unknown model: {model_name}')
    return BaselineCNN(num_classes, channels, float(model_cfg.get('dropout', .3)), int(model_cfg.get('base_channels', 32 if model_name == 'baseline_cnn' else 48)))

