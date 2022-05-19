import os
import sys
from pathlib import Path

# add easy dl to the python path
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
from easydl.metrics import recall_in_k_self_retrieval
from easydl.batch_processing import batch_process_x_y_dataset_and_concat

class Identity(torch.nn.Module):
    def __init__(self, dim=1):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':

    model = convnext_base(pretrained=True, in_22k=False, num_classes=1000) 
    # replace the head with a linear embedder
    model.head = Identity()

    model.to('cuda:0')
    model.eval()
    
    cub_exp = CubMetricLearningExperiment()

    cub_exp.train_ds.change_image_transform(resnet_transform_test)
    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    # evaluate on training set

    ztrain, _ = batch_process_x_y_dataset_and_concat(cub_exp.train_ds, model, batch_size=64)

    print('training', ztrain.shape, )
    mag = ztrain.norm(p=2, dim=1)
    print('mag shape', mag.shape, mag.mean())
    print('mag of norm=1', ztrain.norm(p=1, dim=1).mean())

    ztest, _ = batch_process_x_y_dataset_and_concat(cub_exp.test_ds, model, batch_size=64)
    print('testing', 'shape', ztest.shape)

    mag = ztest.norm(p=2, dim=1)
    print('mag shape', mag.shape, mag.mean())
    print('mag of norm=1', ztest.norm(p=1, dim=1).mean())


