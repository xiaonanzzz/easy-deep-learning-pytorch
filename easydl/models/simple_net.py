import torch
from torch import nn


class SimpleNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(SimpleNet, self).__init__()
        self.channels = 64
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, ),
            nn.Conv2d(self.channels, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)     # batch, channels, 1, 1
        x = self.classifier(x)
        if self.training:
            return x
        else:
            return torch.argmax(x, dim=1)

class SimpleNetV2(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(SimpleNetV2, self).__init__()
        self.channels = 128
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, ),
            nn.Conv2d(self.channels, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)     # batch, channels, 1, 1
        x = self.classifier(x)
        if self.training:
            return x
        else:
            return torch.argmax(x, dim=1)


def get_model_by_name(name):
    return globals()[name]