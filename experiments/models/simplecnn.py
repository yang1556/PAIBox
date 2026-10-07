import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    """Three conv blocks (32x32 -> 16x16 -> 8x8 -> 4x4) + linear head."""

    def __init__(self, num_classes=10, width=64):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        self.block1 = self._make_block(3, c1)
        self.block2 = self._make_block(c1, c2)
        self.block3 = self._make_block(c2, c3)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c3 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _make_block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        return self.head(x)
