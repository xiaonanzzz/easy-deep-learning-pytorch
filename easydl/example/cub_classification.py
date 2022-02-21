from easydl.datasets.cub import CubMetricLearningExperiment, CubClassificationExperiment
from easydl.config import TrainingConfig, RuntimeConfig, ConfigBase, get_config_from_cmd
from easydl.experiments import WandbExperiment
from easydl.image_transform import resnet_transform_train, resnet_transform_test
from easydl.image_model import get_pytorch_model
from easydl.mlp_model import EmbedderClassifier
from easydl.image_classification import train_image_classification_model_2021_nov
from torchvision.models import resnet50

def cub_image_classification_2021_nov():
    # run configuration first
    run_cfg = RuntimeConfig()
    run_cfg.update_values_from_cmd()
    run_cfg.tags.append('resnet 50')

    # prepare experiments
    wandb_exp = WandbExperiment(run_cfg)

    # prepare configurations
    train_cfg = TrainingConfig(optimizer='sgd', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                               lr_decay_step=10, train_batch_size=64, train_epoch=30)
    train_cfg.model = 'resnet50'
    train_cfg.image_size = 448
    train_cfg.pretrained = True
    train_cfg.update_values_from_cmd()

    cub_exp = CubClassificationExperiment(image_size=train_cfg.image_size)

    classifier = get_pytorch_model(train_cfg.model, cub_exp.n_classes, pretrained=train_cfg.pretrained)

    metric_logger = wandb_exp.metric_logger
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