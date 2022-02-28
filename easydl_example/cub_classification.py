from easydl.datasets.cub import CubMetricLearningExperiment, CubClassificationExperiment
from easydl.config import TrainingConfig, RuntimeConfig, ConfigBase, get_config_from_cmd
from easydl.experiments import MetricLogger
from easydl.image_transform import resnet_transform_train, resnet_transform_test
from easydl.image_model import get_pytorch_model
from easydl.mlp_model import EmbedderClassifier
from easydl.image_classification import train_image_classification_model_2021_nov
from torchvision.models import resnet50

def cub_image_classification_2021_nov():
    # run configuration first
    run_cfg = RuntimeConfig(project_name='cub-classification')
    run_cfg.update_values_from_cmd()
    run_cfg.tags.append('resnet 50')

    # prepare experiments
    metric_logger = MetricLogger(run_cfg)

    # prepare configurations
    train_cfg = TrainingConfig(optimizer='sgd', lr=0.003, weight_decay=1e-4, lr_scheduler_type='cosine',
                               lr_decay_step=10, train_batch_size=32, train_epoch=95, nesterov=True)
    train_cfg.model = 'resnet50'
    train_cfg.image_size = 448
    train_cfg.pretrained = True
    train_cfg.update_values_from_cmd()

    cub_exp = CubClassificationExperiment(image_size=train_cfg.image_size)

    classifier = get_pytorch_model(train_cfg.model, cub_exp.n_classes, pretrained=train_cfg.pretrained)

    metric_logger.update_config(train_cfg.dict())

    # run experiment
    train_image_classification_model_2021_nov(
        classifier, cub_exp.train_ds, train_cfg, run_cfg, metric_logger,
        epoch_end_hook=None, eval_train_ds=False, test_ds=cub_exp.test_ds)


def cub_image_classification_timm_data_aug():
    from timm.data import create_transform
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    # run configuration first
    run_cfg = RuntimeConfig(project_name='cub-classification')
    run_cfg.update_values_from_cmd()
    run_cfg.tags.append('timm-aug')

    # prepare experiments
    metric_logger = MetricLogger(run_cfg)

    # prepare configurations
    train_cfg = TrainingConfig(optimizer='sgd', lr=0.003, weight_decay=1e-4, lr_scheduler_type='cosine',
                               lr_decay_step=10, train_batch_size=32, train_epoch=95, nesterov=True)
    train_cfg.model = 'resnet50'
    train_cfg.image_size = 448
    train_cfg.pretrained = True
    train_cfg.color_jitter = 0.4
    train_cfg.auto_augment = 'rand-m9-mstd0.5-inc1'
    train_cfg.interpolation = 'bicubic'
    train_cfg.random_erase_prob = 0.25
    train_cfg.random_erase_mode = 'pixel'
    train_cfg.random_erase_count = 1
    train_cfg.update_values_from_cmd()

    cub_exp = CubClassificationExperiment(image_size=train_cfg.image_size)
    cub_exp.train_ds.change_image_transform(create_transform(
        input_size=train_cfg.image_size,
        is_training=True,
        color_jitter=train_cfg.color_jitter,
        auto_augment=train_cfg.auto_augment,
        interpolation=train_cfg.interpolation,
        re_prob=train_cfg.random_erase_prob,
        re_mode=train_cfg.random_erase_mode,
        re_count=train_cfg.random_erase_count,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ))

    classifier = get_pytorch_model(train_cfg.model, cub_exp.n_classes, pretrained=train_cfg.pretrained)

    metric_logger.update_config(train_cfg.dict())

    # run experiment
    train_image_classification_model_2021_nov(
        classifier, cub_exp.train_ds, train_cfg, run_cfg, metric_logger,
        epoch_end_hook=None, eval_train_ds=False, test_ds=cub_exp.test_ds)


def cub_image_classification_convnext_model():
    from timm.data import create_transform
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from easydl_example.convnext import convnext_tiny, convnext_base
    import torch
    # run configuration first
    run_cfg = RuntimeConfig(project_name='cub-classification')
    run_cfg.update_values_from_cmd()
    run_cfg.tags.append('timm-aug')

    # prepare experiments
    metric_logger = MetricLogger(run_cfg)

    # prepare configurations
    train_cfg = TrainingConfig(optimizer='sgd', lr=0.003, weight_decay=1e-4, lr_scheduler_type='cosine',
                               lr_decay_step=10, train_batch_size=32, train_epoch=95, nesterov=True)
    train_cfg.model = 'base_22k'
    train_cfg.image_size = 448
    train_cfg.pretrained = True
    train_cfg.color_jitter = 0.4
    train_cfg.auto_augment = 'rand-m9-mstd0.5-inc1'
    train_cfg.interpolation = 'bicubic'
    train_cfg.random_erase_prob = 0.25
    train_cfg.random_erase_mode = 'pixel'
    train_cfg.random_erase_count = 1
    train_cfg.update_values_from_cmd()

    cub_exp = CubClassificationExperiment(image_size=train_cfg.image_size)
    cub_exp.train_ds.change_image_transform(create_transform(
        input_size=train_cfg.image_size,
        is_training=True,
        color_jitter=train_cfg.color_jitter,
        auto_augment=train_cfg.auto_augment,
        interpolation=train_cfg.interpolation,
        re_prob=train_cfg.random_erase_prob,
        re_mode=train_cfg.random_erase_mode,
        re_count=train_cfg.random_erase_count,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ))
    if train_cfg.model == 'tiny':
        classifier = convnext_tiny(pretrained=train_cfg.pretrained)
        classifier.head = torch.nn.Linear(classifier.head.in_features, cub_exp.n_classes)
    elif train_cfg.model == 'base_22k':
        classifier = convnext_base(pretrained=train_cfg.pretrained, in_22k=True)
        classifier.head = torch.nn.Linear(classifier.head.in_features, cub_exp.n_classes)
    elif train_cfg.model == 'base_1k':
        classifier = convnext_base(pretrained=train_cfg.pretrained, in_22k=False)
        classifier.head = torch.nn.Linear(classifier.head.in_features, cub_exp.n_classes)

    metric_logger.update_config(train_cfg.dict())

    # run experiment
    train_image_classification_model_2021_nov(
        classifier, cub_exp.train_ds, train_cfg, run_cfg, metric_logger,
        epoch_end_hook=None, eval_train_ds=False, test_ds=cub_exp.test_ds)

if __name__ == '__main__':
    """
    export PYTHONPATH=$HOME/efs/easy-deep-learning-pytorch

    """
    func_name = get_config_from_cmd('func', 'cub_image_classification_2021_nov')
    print('executing function ...', func_name)
    globals()[func_name]()