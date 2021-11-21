import torch
from torch import nn
from torch.nn import functional

class SimpleConvLayersV1(nn.Module):
    def __init__(self, input_channels, output_channels, channels=64, kernel_size=5, downsample_size=2,
                 padding=None) -> None:
        super(SimpleConvLayersV1, self).__init__()
        if padding is None:
            # else padding is an int
            padding = kernel_size // 2

        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(downsample_size),
            nn.Conv2d(channels, output_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return x


class SimpleNet(nn.Module):

    def __init__(self, num_classes: int = 1000, channels=64, downsample_size=2, kernel_size=5) -> None:
        super(SimpleNet, self).__init__()
        padding = kernel_size // 2
        self.channels = channels
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(downsample_size, ),
            nn.Conv2d(self.channels, self.channels, kernel_size=kernel_size, padding=padding),
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

    def __init__(self, num_classes: int = 1000, channels=64, **kwargs) -> None:
        super(SimpleNetV2, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.feature1 = SimpleConvLayersV1(3, 64, channels=channels, **kwargs)
        self.feature2 = SimpleConvLayersV1(3, 64, channels=channels, **kwargs)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.feature1(x)
        x2 = self.feature2(functional.avg_pool2d(x, 3))
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x = functional.relu(x1 + x2)
        x = torch.flatten(x, 1)     # batch, channels, 1, 1
        x = self.classifier(x)
        if self.training:
            return x
        else:
            return torch.argmax(x, dim=1)

def get_model_by_name(name):
    return globals()[name]