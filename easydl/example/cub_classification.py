from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.config import TrainingConfig, RuntimeConfig, ConfigBase, get_config_from_cmd
from easydl.experiments import WandbExperiment
from easydl.image_transform import resnet_transform_train, resnet_transform_test
from easydl.image_model import get_pytorch_model
from easydl.mlp_model import EmbedderClassifier
from easydl.image_classification import train_image_classification_model_2021_nov
from torchvision.models import resnet50

def resnet_50_classification_loss():

    # prepare configurations
    train_cfg = TrainingConfig(optimizer='sgd', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                               lr_decay_step=10, train_batch_size=120, train_epoch=60)
    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig()
    run_cfg.update_values_from_cmd()
    run_cfg.tags.append('resnet_50_from_scratch')

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()
    wandb_exp = WandbExperiment(run_cfg)

    classifier = get_pytorch_model('resnet50', cub_exp.train_classes, pretrained=True)

    cub_exp.train_ds.change_image_transform(resnet_transform_train)
    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    metric_logger = wandb_exp.metric_logger
    metric_logger.update_config(train_cfg.dict())

    # run experiment
    train_image_classification_model_2021_nov(
        classifier, cub_exp.train_ds, train_cfg, run_cfg, metric_logger,
        epoch_end_hook=None, eval_train_ds=True)

if __name__ == '__main__':
    """
    export PYTHONPATH=$HOME/efs/easy-deep-learning-pytorch

    """
    func_name = get_config_from_cmd('func', 'resnet_50_classification_loss')
    print('executing function ...', func_name)
    globals()[func_name]()