import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import transforms
import easydl
from easydl.config import TrainingConfig, RuntimeConfig
from easydl.trainer.image_classification import train_image_classification_model_2021_nov
from easydl.utils import get_config_from_cmd, update_configs_from_cmd
from easydl.utils import batch_process_x_y_dataset_and_concat
from easydl.utils.experiments import prepare_logger
import torch
from torch import nn
from torch.nn import functional as F
import os

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class BootstrapPooling(nn.Module):
    def __init__(self, bootstrap_ratio=0.5):
        super(BootstrapPooling, self).__init__()

        self.bootstrap_ratio = bootstrap_ratio

    def forward(self, x):
        b, c, h, w = x.shape
        if not self.training:
            return F.adaptive_avg_pool2d(x, (1, 1))

        mask = torch.empty(b, h, w, device=x.device, requires_grad=False).uniform_(0, 1)
        mask = mask <= self.bootstrap_ratio
        mask = mask.unsqueeze(1)
        x = x * mask
        x = x.sum(axis=(2, 3), keepdims=True)
        x = x / (1e-8 + mask.sum(axis=(2, 3), keepdims=True))
        return x


class SimpleNet(nn.Module):

    def __init__(self, num_classes: int = 1000, channels=64, downsample_size=2, kernel_size=5, bootstrap_pooling=0.5) -> None:
        super(SimpleNet, self).__init__()
        padding = kernel_size // 2
        self.channels = channels
        self.feature1 = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.AvgPool2d(downsample_size, )
        self.feature2 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.pooling = BootstrapPooling(bootstrap_pooling)
        self.classifier = nn.Linear(self.channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature1(x)
        x = self.pool1(x)
        x = self.feature2(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)     # batch, channels, 1, 1
        x = self.classifier(x)
        if self.training:
            return x
        else:
            return torch.argmax(x, dim=1)



def train_alpha():
    # get configurations from cmd
    import os
    project_name = get_config_from_cmd('project_name', 'walking-dog')
    data_dir = get_config_from_cmd('data_dir', '~/pytorch_data')
    tags = get_config_from_cmd('tags', 'alpha', convert_to_list=True)
    tags.append('ver-{}'.format(easydl.__version__))
    tags.append('{}'.format(__file__))

    train_cfg = TrainingConfig(optimizer='sgd', lr=0.01, weight_decay=5e-4, lr_scheduler_type='cosine',
                               train_epoch=100, train_batch_size=128)
    train_cfg.update_values_from_cmd()
    run_cfg = RuntimeConfig()
    metric_logger = prepare_logger(wandb_key='auto',
                                   project_name=project_name,
                                   tags=tags,  # modify this accordingly !!!
                                   )
    metric_logger.update_config(train_cfg.__dict__)

    train_ds = CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_ds = CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    print('train data shape', train_ds[0][0].shape, str(train_ds[0][0])[:50])
    print('training code version', easydl.__version__)

    model_config = dict(channels=64, downsample_size=2, kernel_size=5, bootstrap_pooling=0.5)
    update_configs_from_cmd(model_config, )
    model = SimpleNet(10, **model_config)
    metric_logger.update_config(model_config)

    train_image_classification_model_2021_nov(model, train_ds, train_cfg, run_cfg, metric_logger, test_ds=test_ds)

if __name__ == '__main__':

    train_alpha()