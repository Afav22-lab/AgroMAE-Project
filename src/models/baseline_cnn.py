import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    Modelo A (Línea Base): CNN Supervisada tradicional adaptada para 13 canales.
    Diseñada para entrenar desde cero con bajo costo computacional (adecuada para CPU/GPU modesta).
    """
    def __init__(self, in_chans=13, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 64x64 -> 32x32
            nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloque 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloque 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits
