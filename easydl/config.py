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
    def __init__(self, *args, device=None, cpu_workers=2, tqdm_disable=False, infer_batch_size=32, model_dir=None, **kwargs):
        super(RuntimeConfig, self).__init__(*args, **kwargs)
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu_workers = cpu_workers
        self.tqdm_disable = tqdm_disable
        self.infer_batch_size = infer_batch_size
        self.model_dir = model_dir
        self.wandb_dir = '~/wandb-exp'


class TrainingConfig(ConfigBase):
    def __init__(self, *args, optimizer='sgd', lr=0.1, weight_decay=1e-4, momentum=0.9,
                 lr_scheduler_type='step', lr_decay_step=10, lr_decay_gamma=0.5,
                 train_batch_size=30, train_epoch=30,  nesterov=False, **kwargs):
        super(TrainingConfig, self).__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.clip_gradient = 10

        self.lr_scheduler_type = lr_scheduler_type
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma

        # datasets related configurations
        self.train_batch_size = train_batch_size
        self.train_epoch = train_epoch


def get_config_from_cmd(key, default=None, key_type=None, kvdict=None, convert_to_list=False, do_expand_path=False):
    """
    It will return default value or value in the argument commend
    if default is None, key_type should be given
    expand_path: True/False, if it is true, then call expand_path
    """
    if kvdict is not None and key in kvdict:
        return kvdict[key]

    import argparse
    pa = argparse.ArgumentParser(allow_abbrev=False)
    pa.add_argument('--{}'.format(key), type=type(default) if key_type is None else key_type, default=default)
    args = pa.parse_known_args()[0]
    value = args.__dict__[key]
    if convert_to_list:
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