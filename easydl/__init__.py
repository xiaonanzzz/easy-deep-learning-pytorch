__version__ = "2.11"

# 2.11 code refactoring
# 2.10 add l2_norm in simple net embedder
# 2.9 change freezing config and logic, train simple net on CUB
# 2.8 fixing a logic bug in bn_freezing in cnn_embedder, fixing shuffle = False in metric learning
# 2.7 fixing a bug in warmup freezing in metric_learning.py
# 2.6, change get_config_from_cmd

from .common import *
from .experiments import MetricLogger