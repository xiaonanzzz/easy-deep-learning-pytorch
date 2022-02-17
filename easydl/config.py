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
        self.device = parse_torch_device('auto')
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

def parse_torch_device(device_str):
    """ auto, cpu, cuda, cuda:0, ..."""
    if device_str == 'auto':
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device(device_str)

def get_config_from_cmd(key, default=None, do_not_expand_path=False):
    """
    return default value or get from cmd

    """
    import argparse
    pa = argparse.ArgumentParser(allow_abbrev=False)
    if type(default) in [int, float]:
        t = type(default)
    elif type(default) == type(None):
        t = None
    elif type(default) in [str, list, torch.device, bool]:
        t = str
    else:
        raise NotImplementedError('the given default is not supported', default, type(default), key)

    pa.add_argument('--{}'.format(key), type=t, default=default)
    args = pa.parse_known_args()[0]
    value = args.__dict__[key]
    if type(value) == str:
        if len(value) > 0 and (value[0] in ['~', '$']) and not do_not_expand_path:
            value = expand_path(value)
        elif type(default) == list:
            value = value.split(',') if len(value) > 0 else []
        elif type(default) == torch.device:
            value = parse_torch_device(value)
        elif type(default) == bool:
            if value in ['False', 'false', 0]:
                value = False
            else:
                value = True
    if type(value) != type(default):
        print('WARNING!!! type difference: parsed value vs. default ', value, default)
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