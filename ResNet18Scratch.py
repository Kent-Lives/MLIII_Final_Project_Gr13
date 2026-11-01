import torch
from torch import nn
from torch.nn import functional as F

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class BasicResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResNetBlock, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride == 1 and in_channels == out_channels:
            # if the input and output have the same shape, we can use an identity shortcut
            self.skip = nn.Identity()
        else:
            # otherwise, we use a conv shortcut so that we output the right dims
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.conv1(x)) + self.skip(x)
        return self.ReLU(out)

class ResNet18Scratch(nn.Module):
    def __init__(self, layers, num_classes=2):
        super(ResNet18Scratch, self).__init__()
        self.in_channels = 64
        self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64,  layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicResNetBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x