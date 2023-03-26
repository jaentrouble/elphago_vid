import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_n) -> None:
        super().__init__()
        self.layer_n = layer_n
        self.layers = nn.ModuleList()
        for i in range(layer_n):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
            self.layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # residual connection
        residual = x
        for l in self.layers:
            x = self.relu(l(x))
        return x + residual

class ResNet(nn.Module):
    def __init__(self, in_channels, out_classes, block_n, layer_n) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(block_n):
            self.layers.append(ResBlock(in_channels, 64, 3, layer_n))
            self.layers.append(nn.MaxPool2d(2, 2))
            in_channels = 64
        self.layers.append(nn.Conv2d(64, out_classes, 1))
        self.layers.append(nn.AdaptiveAvgPool2d(1))
        self.relu = nn.ReLU()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = x.reshape(x.shape[0], -1)
        return x
    
