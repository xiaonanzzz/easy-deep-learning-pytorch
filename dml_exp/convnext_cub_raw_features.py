import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from modulefinder import Module
from easydl.algorithm.image_classification import train_image_classification_model_2021_nov
from easydl.algorithm.metric_learning import *
from easydl.config import RuntimeConfig, TrainingConfig, get_config_from_cmd
from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.experiments import MetricLogger
from easydl.image_transform import resnet_transform_test, resnet_transform_train, timm_image_transform_imagenet_default
from easydl.models.image_model import Resnet50PALVersion
from easydl.models.mlp_model import EmbedderClassifier, LinearEmbedder, L2Normalization
from easydl.models.convnext import convnext_base, get_convnext_version_augmentation_config


if __name__ == '__main__':

    model = convnext_base(pretrained=True, in_22k=True, num_classes=21841) 
    # replace the head with a linear embedder
    model.head = L2Normalization()

    model.to('cuda:0')
    model.eval()
    
    cub_exp = CubMetricLearningExperiment()

    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    recall_at_k = cub_exp.evaluate_model(model)
    
    print(recall_at_k)

