from easydl.datasets.cars196 import Cars196MetricLearningExp
from easydl.algorithm.metric_learning import *
from easydl.config import TrainingConfig, RuntimeConfig, get_config_from_cmd
from easydl.experiments import MetricLogger
from easydl.image_transform import resnet_transform_train, resnet_transform_test, timm_image_transform_imagenet_default
from easydl.models.image_model import Resnet50PALVersion
from easydl.models.mlp_model import EmbedderClassifier, LinearEmbedder
from easydl.algorithm.image_classification import train_image_classification_model_2021_nov

if __name__ == '__main__':
    from easydl.models.convnext import convnext_base, get_convnext_version_augmentation_config
    # prepare configurations
    train_cfg = TrainingConfig(optimizer='adamw', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                                lr_decay_step=5, train_batch_size=120, train_epoch=60)
    train_cfg.pretrained = True
    train_cfg.from_other(get_convnext_version_augmentation_config())

    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig(project_name='dml_cars_convnext')
    run_cfg.update_values_from_cmd()
    algo_cfg = ProxyAnchorLossConfig()
    algo_cfg.update_values_from_cmd()

    model_cfg = dict()

    # prepare experiments
    cars_exp = Cars196MetricLearningExp()

    metric_logger = MetricLogger(run_cfg)

    # prepare model
    # because it's pre-trained on 22k, so, set num_classes = 21841
    model = convnext_base(pretrained=train_cfg.pretrained, in_22k=True, num_classes=21841)
    # replace the head with a linear embedder
    model.head = LinearEmbedder(model.head.in_features, algo_cfg.embedding_size)
    freezing_params = list(set(model.parameters()) - set(model.head.parameters()))

    train_ds = cars_exp.train_ds(timm_image_transform_imagenet_default(train_cfg))

    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())
    metric_logger.update_config(model_cfg)

    def epoch_end(**kwargs):
        print('evaluting the model on testing data...')
        recall_at_k = cars_exp.evaluate_model(model, image_transform=resnet_transform_test)
        metric_logger.log({'recall@{}'.format(k): v for k, v in recall_at_k.items()})

    # run experiment
    train_embedding_model_with_proxy_anchor_loss_with_warmup_freeze(model, train_ds, cars_exp.train_classes,
                                                    metric_logger, train_cfg, run_cfg, algo_cfg,
                                                                    epoch_end_hook=epoch_end,
                                                                    freezing_params_during_warmup=freezing_params)