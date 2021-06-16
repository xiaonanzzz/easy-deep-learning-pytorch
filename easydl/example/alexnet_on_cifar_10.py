import pandas as pd
import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.models import GoogLeNet
from torchvision.transforms import ToTensor
from easydl.trainer.image_classification import ImageClassificationTrainer
from torch import nn

class SimpleNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(SimpleNet, self).__init__()
        self.channels = min(256, num_classes * 5)
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Dropout2d(),
            nn.Conv2d(self.channels, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)     # batch, channels, 1, 1
        return x

if __name__ == '__main__':
    import os
    print('working directory', os.getcwd())

    train_data = CIFAR10('tmp/data', train=True, download=True, transform=ToTensor())
    print('data shape', train_data[0][0].shape)
    model = SimpleNet(10)

    trainer = ImageClassificationTrainer()

    trainer.train(model, train_data, epoch_end_hook=None)

