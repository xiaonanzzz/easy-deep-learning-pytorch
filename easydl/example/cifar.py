from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

import easydl
from easydl.config import TrainingConfig, RuntimeConfig
from easydl.models.simple_net import get_model_by_name
from easydl.trainer.image_classification import train_image_classification_model_2021_nov
from easydl.utils import get_config_from_cmd
from easydl.utils.experiments import prepare_logger


def get_cifar_image_transformer():
    return Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])])


def train_simple_net():
    # get configurations from cmd
    project_name = get_config_from_cmd('project_name', 'cifar_10')
    wandb_key = get_config_from_cmd('wandb_key', None, key_type=str)
    data_dir = get_config_from_cmd('data_dir', '~/pytorch_data')
    tags = get_config_from_cmd('tags', 'cifar10,simple-net').split(',')
    model_name = get_config_from_cmd('model_name', 'SimpleNetV2')

    tags.append('ver-{}'.format(easydl.__version__))
    tags.append('model-name-{}'.format(model_name))

    train_ds = CIFAR10(data_dir, train=True, download=True, transform=get_cifar_image_transformer())
    test_ds = CIFAR10(data_dir, train=False, download=True, transform=get_cifar_image_transformer())

    print('train data shape', train_ds[0][0].shape, str(train_ds[0][0])[:50])
    model_cls = get_model_by_name(model_name)
    model = model_cls(10)

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

