from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import transforms
import easydl
from easydl.config import TrainingConfig, RuntimeConfig, get_config_from_cmd, update_configs_from_cmd
from easydl.simple_net import SimpleNet, SimpleNetV2
from easydl.image_classification import train_image_classification_model_2021_nov
from easydl.experiments import prepare_logger


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

def get_cifar_image_transformer():
    return Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])])


def train_cifar():
    # get configurations from cmd
    project_name = get_config_from_cmd('project_name', 'cifar_10')
    wandb_key = get_config_from_cmd('wandb_key', None, value_type=str)
    data_dir = get_config_from_cmd('data_dir', '~/pytorch_data')
    model_name = get_config_from_cmd('model', 'resnet18-si')

    tags = get_config_from_cmd('tags', '', convert_to_list=True)
    tags.append('ver-{}'.format(easydl.__version__))
    tags.append(model_name)

    train_cfg = TrainingConfig(optimizer='sgd', lr=0.01, weight_decay=5e-4, lr_scheduler_type='cosine',
                               train_epoch=100, train_batch_size=128)
    train_cfg.update_values_from_cmd()
    run_cfg = RuntimeConfig()
    metric_logger = prepare_logger(wandb_key=wandb_key,
                                   project_name=project_name,
                                   tags=tags,  # modify this accordingly !!!
                                   )
    metric_logger.update_config(train_cfg.__dict__)

    train_ds = CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_ds = CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    print('train datasets shape', train_ds[0][0].shape, str(train_ds[0][0])[:50])
    print('training code version', easydl.__version__)

    if model_name == 'resnet18-si':
        from easydl.resnet_small_image import ResNet18
        model = ResNet18(10)
    if model_name == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained=False, num_classes=10)

    if model_name == 'simplenet':
        model_config = dict(channels=64, downsample_size=2, kernel_size=5)
        update_configs_from_cmd(model_config, )
        model = SimpleNet(10, **model_config)
        metric_logger.update_config(model_config)
    if model_name == 'simplenetv2':
        model_config = dict(channels=64, downsample_size=2, kernel_size=5)
        update_configs_from_cmd(model_config, )
        model = SimpleNetV2(10, **model_config)
        metric_logger.update_config(model_config)

    train_image_classification_model_2021_nov(model, train_ds, train_cfg, run_cfg, metric_logger, test_ds=test_ds)


if __name__ == '__main__':
    func_name = get_config_from_cmd('function', 'train_cifar')
    # run function
    globals()[func_name]()

