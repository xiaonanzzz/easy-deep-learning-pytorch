from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.algorithm.metric_learning import *
from easydl.config import TrainingConfig, RuntimeConfig, get_config_from_cmd
from easydl.experiments import MetricLogger
from easydl.image_transform import resnet_transform_train, resnet_transform_test
from easydl.models.image_model import Resnet50PALVersion
from easydl.models.mlp_model import EmbedderClassifier, LinearEmbedder
from easydl.algorithm.image_classification import train_image_classification_model_2021_nov


def proxy_anchor_loss_paper():
    """
    reproducing the result from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 120 \
                --lr 1e-4 \
                --dataset cub \
                --warm 5 \
                --bn-freeze 1 \
                --lr-decay-step 5

    :return:
    """
    # prepare configurations
    train_cfg = TrainingConfig(optimizer='adamw', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                               lr_decay_step=5, train_batch_size=120, train_epoch=60)
    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig()
    run_cfg.update_values_from_cmd()
    algo_cfg = ProxyAnchorLossConfig()
    algo_cfg.update_values_from_cmd()

    model_cfg = dict(pretrained=True, bn_freeze=True)

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()
    metric_logger = MetricLogger(run_cfg)
    model = Resnet50PALVersion(algo_cfg.embedding_size, **model_cfg)

    cub_exp.train_ds.change_image_transform(resnet_transform_train)
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
                                                                    freezing_params_during_warmup=model.get_pretrained_parameters())

def resnet_50_from_scratch():
    # prepare configurations
    train_cfg = TrainingConfig(optimizer='sgd', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                               lr_decay_step=10, train_batch_size=120, train_epoch=60)
    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig()
    run_cfg.update_values_from_cmd()
    run_cfg.tags.append('resnet_50_from_scratch')

    algo_cfg = ProxyAnchorLossConfig()
    algo_cfg.update_values_from_cmd()

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()
    metric_logger = MetricLogger(run_cfg)

    model = Resnet50PALVersion(algo_cfg.embedding_size, pretrained=False, bn_freeze=False)

    cub_exp.train_ds.change_image_transform(resnet_transform_train)
    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())

    def epoch_end(**kwargs):
        print('evaluting the model on testing data...')
        recall_at_k = cub_exp.evaluate_model(model)
        metric_logger.log({'recall@{}'.format(k): v for k, v in recall_at_k.items()})

    # run experiment
    train_embedding_model_with_proxy_anchor_loss_with_warmup_freeze(model, cub_exp.train_ds, cub_exp.train_classes,
                                                    metric_logger, train_cfg, run_cfg, algo_cfg, epoch_end_hook=epoch_end,
                                                                    freezing_params_during_warmup=None)

def resnet_50_clf_loss_v1():
    # prepare configurations
    run_cfg = RuntimeConfig()
    run_cfg.update_values_from_cmd()
    run_cfg.tags.append('resnet_50_clf_loss_v1')
    metric_logger = MetricLogger(run_cfg)

    train_cfg = TrainingConfig(optimizer='sgd', lr=1e-4, weight_decay=1e-5, lr_scheduler_type='cosine',
                               train_batch_size=120, train_epoch=60)
    train_cfg.pretrained = True
    train_cfg.is_norm = True
    train_cfg.bn_freeze = True
    train_cfg.update_values_from_cmd()

    algo_cfg = ProxyAnchorLossConfig()
    algo_cfg.update_values_from_cmd()

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()

    embedder = Resnet50PALVersion(algo_cfg.embedding_size,
                                  pretrained=train_cfg.pretrained, is_norm=train_cfg.is_norm, bn_freeze=train_cfg.bn_freeze)
    embedder_classifier = EmbedderClassifier(embedder, algo_cfg.embedding_size, cub_exp.train_classes)

    cub_exp.train_ds.change_image_transform(resnet_transform_train)
    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())

    def epoch_end(**kwargs):
        print('evaluting the model on testing data...')
        embedder.is_norm = True
        recall_at_k = cub_exp.evaluate_model(embedder)
        embedder.is_norm = train_cfg.is_norm
        metric_logger.log({'recall@{}'.format(k): v for k, v in recall_at_k.items()})

    # run experiment
    train_image_classification_model_2021_nov(
        embedder_classifier, cub_exp.train_ds, train_cfg, run_cfg, metric_logger,
        epoch_end_hook=epoch_end)


def multi_loss_v1():
    # prepare configurations
    train_cfg = TrainingConfig(optimizer='adamw', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='cosine',
                               lr_decay_step=5, train_batch_size=120, train_epoch=60)
    train_cfg.is_norm = False
    train_cfg.pretrained = True
    train_cfg.bn_freeze = True
    train_cfg.update_values_from_cmd()


    run_cfg = RuntimeConfig(project_name='cub-dml')
    run_cfg.update_values_from_cmd()
    algo_cfg = ProxyAnchorLossConfig()
    algo_cfg.update_values_from_cmd()

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()
    metric_logger = MetricLogger(run_cfg)
    model = Resnet50PALVersion(algo_cfg.embedding_size, pretrained=train_cfg.pretrained, is_norm=train_cfg.is_norm, bn_freeze=train_cfg.bn_freeze)
    
    cub_exp.train_ds.change_image_transform(resnet_transform_train)
    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())

    def epoch_end(**kwargs):
        print('evaluting the model on testing data...')
        recall_at_k = cub_exp.evaluate_model(model)
        metric_logger.log({'recall@{}'.format(k): v for k, v in recall_at_k.items()})

    # run experiment
    train_embedder_proxy_anchor_loss_and_classification_loss(model, cub_exp.train_ds, cub_exp.train_classes,
                                                             metric_logger, train_cfg, run_cfg, algo_cfg,
                                                             epoch_end_hook=epoch_end,
                                                             freezing_params_during_warmup=model.get_pretrained_parameters())


def convnext_exp():
    """
    reproducing the result from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 120 \
                --lr 1e-4 \
                --dataset cub \
                --warm 5 \
                --bn-freeze 1 \
                --lr-decay-step 5

    :return:
    """
    from easydl.models.convnext import convnext_base
    # prepare configurations
    train_cfg = TrainingConfig(optimizer='adamw', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                               lr_decay_step=5, train_batch_size=120, train_epoch=60)
    train_cfg.pretrained = True
    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig(project_name='convnext_cub_debug')
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
    
    cub_exp.train_ds.change_image_transform(resnet_transform_train)
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
    """
    export PYTHONPATH=$HOME/efs/easy-deep-learning-pytorch

    python3 -m easydl.example.proxy_anchor_loss_cub --project_name cub-dml
    """
    func_name = get_config_from_cmd('func', 'multi_loss_v1')
    print('executing function ...', func_name)
    globals()[func_name]()

