from easydl.algorithm.image_classification import train_image_classification_model_2021_nov
from easydl.algorithm.metric_learning import *
from easydl.config import RuntimeConfig, TrainingConfig, get_config_from_cmd
from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.experiments import MetricLogger
from easydl.image_transform import resnet_transform_test, resnet_transform_train, timm_image_transform_imagenet_default
from easydl.models.image_model import Resnet50PALVersion
from easydl.models.mlp_model import EmbedderClassifier, LinearEmbedder
from easydl.models.convnext import convnext_base, get_convnext_version_augmentation_config


def convnext_exp():
    """
    reproducing the result from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    """


    # prepare configurations
    train_cfg = TrainingConfig(optimizer='adamw', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                               lr_decay_step=5, train_batch_size=120, train_epoch=60)
    train_cfg.pretrained = True
    train_cfg.from_other(get_convnext_version_augmentation_config())

    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig(project_name='convnext_cub')
    run_cfg.update_values_from_cmd()
    
    algo_cfg = ProxyAnchorLossConfig()
    algo_cfg.update_values_from_cmd()

    model_cfg = dict()

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()

    metric_logger = MetricLogger(run_cfg)

    # prepare model
    # because it's pre-trained on 22k, so, set num_classes = 21841
    model = convnext_base(pretrained=train_cfg.pretrained, in_22k=True, num_classes=21841)
    # replace the head with a linear embedder
    model.head = LinearEmbedder(model.head.in_features, algo_cfg.embedding_size)
    freezing_params = list(set(model.parameters()) - set(model.head.parameters()))
    
    cub_exp.train_ds.change_image_transform(timm_image_transform_imagenet_default(train_cfg))
    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())
    metric_logger.update_config(model_cfg)

    def epoch_end(**kwargs):
        print('evaluting the model on testing data...')
        recall_at_k = cub_exp.evaluate_model(model)
        metric_logger.log({'recall@{}'.format(k): v for k, v in recall_at_k.items()})

    # run experiment
    train_embedding_model_with_proxy_anchor_loss_with_warmup_freeze(model, cub_exp.train_ds, cub_exp.train_classes,
                                                    metric_logger, train_cfg, run_cfg, algo_cfg,
                                                                    epoch_end_hook=epoch_end,
                                                                    freezing_params_during_warmup=freezing_params)

if __name__ == '__main__':
    convnext_exp()

