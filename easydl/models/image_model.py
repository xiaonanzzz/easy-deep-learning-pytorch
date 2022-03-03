import torch
from torch import nn
from torch.nn import functional
from torchvision.models.resnet import resnet50
import torch.nn.init as init

from easydl.models.mlp_model import LinearEmbedder


def get_pytorch_model(name, num_classes, pretrained=True):
    from torchvision.models import resnet50
    if name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, num_classes, bias=True)
        return model



class Resnet50PALVersion(nn.Module):
    """
    https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/net/resnet.py
    """
    def __init__(self, embedding_size, pretrained=True, is_norm=True, bn_freeze=True):
        super(Resnet50PALVersion, self).__init__()

        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        self.bn_freeze = bn_freeze
        self.pretrained = pretrained


    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)

        if self.is_norm:
            x = self.l2_norm(x)

        return x

    def train(self, mode: bool = True):
        super(Resnet50PALVersion, self).train(mode=mode)

        # if bn_freeze and training mode, override the bn mode.
        if self.bn_freeze and mode:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)


    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

    def get_pretrained_parameters(self):
        return list(set(self.parameters()) - set(self.model.embedding.parameters()))


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


class SimpleNetEmbedder(nn.Module):

    def __init__(self, embedding_size=32, channels=64, downsample_size=2, kernel_size=5) -> None:
        super(SimpleNetEmbedder, self).__init__()
        padding = kernel_size // 2
        self.channels = channels
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.features = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(downsample_size, ),
            nn.Conv2d(self.channels, self.channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.embedder = LinearEmbedder(channels, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)     # batch, channels, 1, 1
        x = self.embedder(x)
        return x


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