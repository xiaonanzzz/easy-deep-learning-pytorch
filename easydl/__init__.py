__version__ = "2.8"

# 2.8 fixing a logic bug in bn_freezing in cnn_embedder, fixing shuffle = False in metric learning
# 2.7 fixing a bug in warmup freezing in metric_learning.py
# 2.6, change get_config_from_cmd

from .common import *
from .experiments import MetricLogger