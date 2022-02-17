from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.metric_learning import ProxyAnchorLossConfig, train_embedding_model_with_proxy_anchor_loss_v2
from easydl.config import TrainingConfig, RuntimeConfig
from easydl.experiments import WandbExperiment


def resnet50_example_no_pretrain_proxy_anchor_loss_paper():
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
    import os
    from easydl.cnn_embedder import Resnet50PALVersion
    # prepare configurations
    train_cfg = TrainingConfig(optimizer='sgd', lr=1e-4, weight_decay=1e-4, momentum=0.9,
              lr_scheduler_type='cosine', train_batch_size=10, train_epoch=30, nesterov=False)
    train_cfg.update_values_from_cmd()
    run_cfg = RuntimeConfig()
    run_cfg.update_values_from_cmd()
    algo_cfg = ProxyAnchorLossConfig()
    algo_cfg.update_values_from_cmd()

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()
    wandb_exp = WandbExperiment(run_cfg)
    model = Resnet50PALVersion(algo_cfg.embedding_size, pretrained=False, bn_freeze=False)
    metric_logger = wandb_exp.metric_logger
    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())

    def epoch_end(**kwargs):
        print('evaluting the model on testing data...')
        recall_at_k = cub_exp.evaluate_model(model)
        metric_logger.log({'recall@{}'.format(k): v for k, v in recall_at_k.items()})

    # run experiment
    train_embedding_model_with_proxy_anchor_loss_v2(model, cub_exp.trainds, cub_exp.train_classes,
                                                    metric_logger, train_cfg, run_cfg, algo_cfg, epoch_end_hook=epoch_end)


if __name__ == '__main__':
    resnet50_example_no_pretrain_proxy_anchor_loss_paper()

