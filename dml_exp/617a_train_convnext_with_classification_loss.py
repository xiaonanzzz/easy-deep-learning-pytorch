import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from modulefinder import Module
from easydl.algorithm.metric_learning import *
from easydl.algorithm.losses import *
from easydl.config import RuntimeConfig, TrainingConfig, get_config_from_cmd
from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.experiments import MetricLogger
from easydl.image_transform import resnet_transform_test, resnet_transform_train, timm_image_transform_imagenet_default
from easydl.models.image_model import Resnet50PALVersion
from easydl.models.mlp_model import EmbedderClassifier, LinearEmbedder, L2Normalization, Identity
from easydl.models.convnext import convnext_base, get_convnext_version_augmentation_config
from easydl.algorithm.image_classification import train_image_classification_model_2021_nov

import os

def convnext_exp():

    # prepare configurations
    train_cfg = TrainingConfig(optimizer='adamw', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                               lr_decay_step=5, train_batch_size=50, train_epoch=60, warmup_epoch=1)
    train_cfg.pretrained = True
    train_cfg.from_other(get_convnext_version_augmentation_config())
    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig(project_name='convnext_cub')
    run_cfg.update_values_from_cmd()
    
    algo_cfg = ProxyAnchorLossConfig(embedding_size=1024)
    algo_cfg.update_values_from_cmd()

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()

    metric_logger = MetricLogger(run_cfg)

    # prepare model
    # because it's pre-trained on 22k, so, set num_classes = 21841
    model = convnext_base(pretrained=train_cfg.pretrained, in_22k=True, num_classes=21841) 
    # replace the head with a linear embedder
    model.head = Identity()

    embedder_classifier = EmbedderClassifier(model, 1024, cub_exp.train_classes)
    
    cub_exp.train_ds.change_image_transform(timm_image_transform_imagenet_default(train_cfg))
    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())

    def epoch_end(**kwargs):
        print('evaluting the model on testing data...')
        recall_at_k = cub_exp.evaluate_model(model)
        metric_logger.log({'recall@{}'.format(k): v for k, v in recall_at_k.items()})

        # save model 
        local_path = os.path.join(metric_logger.local_run_path, 'last_model.torch')
        torch.save(model.state_dict(), local_path)
        if metric_logger.get_best_step('recall@1') == metric_logger.current_step:
            metric_logger.set_summary('best/step', metric_logger.current_step)
            metric_logger.set_summary('best/recall@1', metric_logger.metrics['recall@1'][metric_logger.current_step])

            best_path = os.path.join(metric_logger.local_run_path, 'best_model.torch')
            torch.save(model.state_dict(), best_path)


    # run model training
     # run experiment
    train_image_classification_model_2021_nov(
        embedder_classifier, cub_exp.train_ds, train_cfg, run_cfg, metric_logger,
        epoch_end_hook=epoch_end)

if __name__ == '__main__':
    convnext_exp()

