__version__ = "3.3"


"""
3.3 add deep fashion inshop dataset
3.2 code refactoring, add ConvNeXt model
3.1 move examples to easydl_example
3.0 code refactor

"""


"""
2.19 try multi loss in dml
2.18 refactor metric logger
2.17 add config of lr_min for cosine scheduler
2.16 debuging image classification module, fixed the issue of no shuffle in training...
2.15 cub classification, 200 classes, 
2.14 refactor cifar code, add cub_classification on first 100 CUB classes
2.13 train resnet 50 model from scratch using classification loss
2.12 change cub default transform, train resnet from scratch on CUB metric learning
2.11 code refactoring
2.10 add l2_norm in simple net embedder
2.9 change freezing config and logic, train simple net on CUB
2.8 fixing a logic bug in bn_freezing in cnn_embedder, fixing shuffle = False in metric learning
2.7 fixing a bug in warmup freezing in metric_learning.py
2.6, change get_config_from_cmd
"""

from .config import *
from .common import *
from .experiments import MetricLogger



