"""
Layout must stay as-is, because the runner relies on it:
  - class NiN(num_classes=...) with an nn.Sequential called `sequential`
  - model.sequential[6] is the output NiNBlock, which has its own `sequential` ending in nn.ReLU
    (the runner swaps that last ReLU for nn.Identity so the class scores are not clamped at 0)

Shapes for a 224x224 input:
  NiNBlock(3->96, k11, s4)       -> 96 x 54 x 54
  MaxPool(3, 2)                  -> 96 x 26 x 26
  NiNBlock(96->256, k5, p2)      -> 256 x 26 x 26
  MaxPool(3, 2)                  -> 256 x 12 x 12
  NiNBlock(256->384, k3, p1)     -> 384 x 12 x 12
  MaxPool(3, 2)                  -> 384 x 5 x 5
  NiNBlock(384->num_classes, k3) -> num_classes x 5 x 5
  AdaptiveAvgPool(1, 1)          -> num_classes x 1 x 1   (global average pooling, no dense layer)
  flatten                        -> (batch, num_classes)
"""
import torch
import torch.nn as nn


class NiNBlock(nn.Module):
    """One spatial convolution followed by two 1x1 convolutions (the "micro network"),
    each with batch norm and ReLU. Convolutions have no bias because batch norm follows."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NiNBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),          # in the output block, the runner replaces this one
        ]
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class NiN(nn.Module):
    def __init__(self, num_classes=2):
        super(NiN, self).__init__()
        layers = [
            NiNBlock(3, 96, kernel_size=11, stride=4, padding=0),           # 0
            nn.MaxPool2d(kernel_size=3, stride=2),                          # 1
            NiNBlock(96, 256, kernel_size=5, stride=1, padding=2),          # 2
            nn.MaxPool2d(kernel_size=3, stride=2),                          # 3
            NiNBlock(256, 384, kernel_size=3, stride=1, padding=1),         # 4
            nn.MaxPool2d(kernel_size=3, stride=2),                          # 5
            NiNBlock(384, num_classes, kernel_size=3, stride=1, padding=1), # 6  output block
            nn.AdaptiveAvgPool2d((1, 1)),                                   # 7  global average pooling
        ]
        self.sequential = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init for convolutions (suits ReLU), batch norm starts as the identity."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        intermediate = self.sequential(x)
        return torch.flatten(intermediate, start_dim=1)      # (batch, num_classes) raw scores


if __name__ == "__main__":
    model = NiN(num_classes=2)
    out = model(torch.randn(2, 3, 224, 224))
    print("output shape:", tuple(out.shape))                 # expected (2, 2)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
