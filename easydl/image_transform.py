from torchvision import transforms
from easydl import ImageAugmentationConfig

resnet_mean = [0.485, 0.456, 0.406]
resnet_std = [0.229, 0.224, 0.225]


def make_transform_train_v1(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

def make_transform_test_v1(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])


resnet_transform_train = make_transform_train_v1(image_size=224)
resnet_transform_test = make_transform_test_v1(image_size=224)


def timm_image_transform_imagenet_default(aug_cfg: ImageAugmentationConfig, is_training=True):
    from timm.data import create_transform
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    t = create_transform(
        input_size=aug_cfg.image_size,
        is_training=is_training,
        color_jitter=aug_cfg.color_jitter,
        auto_augment=aug_cfg.timm_auto_augment,
        interpolation=aug_cfg.interpolation,
        re_prob=aug_cfg.random_erase_prob,
        re_mode=aug_cfg.random_erase_mode,
        re_count=aug_cfg.random_erase_count,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return t