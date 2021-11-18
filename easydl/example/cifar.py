import torch
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

import easydl
from easydl.config import TrainingConfig, RuntimeConfig
from easydl.trainer.image_classification import train_image_classification_model_2021_nov
from easydl.utils import get_config_from_cmd
from easydl.utils.experiments import prepare_logger


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

def get_cifar_image_transformer():
    return Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])])


def train_simple_net():
    # get configurations from cmd
    project_name = get_config_from_cmd('project_name', 'cifar_10')
    wandb_key = get_config_from_cmd('wandb_key', None, key_type=str)
    data_dir = get_config_from_cmd('data_dir', '~/pytorch_data')
    tags = get_config_from_cmd('tags', 'cifar10,simple-net').split(',')
    tags.append('ver-{}'.format(easydl.__version__))

    train_ds = CIFAR10(data_dir, train=True, download=True, transform=get_cifar_image_transformer())
    test_ds = CIFAR10(data_dir, train=False, download=True, transform=get_cifar_image_transformer())

    print('train data shape', train_ds[0][0].shape, str(train_ds[0][0])[:50])
    model = SimpleNet(10)

    train_cfg = TrainingConfig(optimizer='sgd', lr=0.1)
    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig()
    metric_logger = prepare_logger(wandb_key=wandb_key,
                                   project_name=project_name,
                                   tags=tags,    # modify this accordingly !!!
                                   )
    metric_logger.update_config(train_cfg.__dict__)
    print('training code version', easydl.__version__)
    train_image_classification_model_2021_nov(model, train_ds, train_cfg, run_cfg, metric_logger, test_ds=test_ds)


def train_resnet_18():
    # get configurations from cmd
    project_name = get_config_from_cmd('project_name', 'cifar_10')
    wandb_key = get_config_from_cmd('wandb_key', None, key_type=str)
    data_dir = get_config_from_cmd('data_dir', '~/pytorch_data')
    tags = get_config_from_cmd('tags', 'cifar10,resnet-18').split(',')
    tags.append('ver-{}'.format(easydl.__version__))

    train_ds = CIFAR10(data_dir, train=True, download=True, transform=get_cifar_image_transformer())
    test_ds = CIFAR10(data_dir, train=False, download=True, transform=get_cifar_image_transformer())

    print('train data shape', train_ds[0][0].shape, str(train_ds[0][0])[:50])
    from torchvision.models import resnet18
    model = resnet18(pretrained=False, num_classes=10)

    train_cfg = TrainingConfig(optimizer='sgd')
    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig()
    metric_logger = prepare_logger(wandb_key=wandb_key,
                                   project_name=project_name,
                                   tags=tags,    # modify this accordingly !!!
                                   )
    metric_logger.update_config(train_cfg.__dict__)
    print('training code version', easydl.__version__)
    train_image_classification_model_2021_nov(model, train_ds, train_cfg, run_cfg, metric_logger, test_ds=test_ds)

if __name__ == '__main__':
    func_name = get_config_from_cmd('function', 'train_resnet_18')
    # run function
    globals()[func_name]()

