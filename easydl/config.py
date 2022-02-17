import torch

from easydl import expand_path

"""
This module provides helps in defining, using and managing configurations. It serves the following purpose:
1) configurations can be stored in python object, so that coding tools can auto-complete and coders don't need to memorize them
2) every algorithm has different default or suggested configurations, therefore, each algorithm can define their default setting
3) for users/coders, they only need to modify a little bit accordingly

ConfigBase are base class to define some common configurations.

"""
class ConfigBase(object):
    def __init__(self, *args, **kwargs):
        super(ConfigBase, self).__init__(*args, **kwargs)

    def dict(self):
        return self.__dict__.copy()

    def from_dict(self, kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def update_values_from_cmd(self, prefix=''):
        update_configs_from_cmd(self.__dict__, prefix=prefix)


def merging_configs(*configs: [ConfigBase]):
    cfg = ConfigBase()
    for c in configs:
        cfg.from_dict(c.dict())
    return cfg

class RuntimeConfig(ConfigBase):
    def __init__(self, *args, **kwargs):
        self.device = None
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu_workers = 4
        self.tqdm_disable = 0
        self.infer_batch_size = 128
        self.model_dir = ''
        self.project_name = 'debug'
        self.wandb_dir = '~/wandb-exp'
        self.tags = []

        self.from_dict(kwargs)


class TrainingConfig(ConfigBase):
    def __init__(self, *args, **kwargs):
        self.optimizer = 'sgd'  # [adamw, adam, rmsprop]
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.nesterov = False
        self.clip_gradient = 10

        self.warmup_epoch = 1
        self.lr_scheduler_type = 'step' # cosine
        self.lr_decay_step = 10
        self.lr_decay_gamma = 0.5

        # datasets related configurations
        self.train_batch_size = 128
        self.train_epoch = 30

        # read from kwargs
        self.from_dict(kwargs)


def get_config_from_cmd(key, default=None, value_type=None, convert_to_list=False, do_expand_path=False):
    """
    It will return default value or value in the argument commend
    if default is None, key_type should be given
    expand_path: True/False, if it is true, then call expand_path
    """

    import argparse
    pa = argparse.ArgumentParser(allow_abbrev=False)
    pa.add_argument('--{}'.format(key), type=value_type, default=default)
    args = pa.parse_known_args()[0]
    value = args.__dict__[key]
    if convert_to_list or type(default) == list:
        value = value.split(',') if len(value) > 0 else []
    if type(value) == str and do_expand_path:
        return expand_path(value)
    return value


def update_configs_from_cmd(config_dict, prefix=''):
    """
    config_dict: {key: value} | object (__dict__ will be used)
    prefix: str, it's used to concat the key from arguments, for de-duplicating purpose. e.g., the original key is 'lr'
            and the prefix is 'train-', the key used in args should be 'train-lr'.

    """
    if isinstance(config_dict, dict):
        config_dict = config_dict
    else:
        config_dict = config_dict.__dict__

    for key, value in config_dict.items():
        value1 = get_config_from_cmd('{}{}'.format(prefix, key), default=value)
        config_dict[key] = value1