import torch

import random
import numpy as np
import time
import datetime
import os


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.data.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def prepare_path(path):
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)


def expand_path(path):
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return path

def save_model(model, path):
    """ create the dirs for the model path, then save the model"""
    prepare_path(path)
    torch.save(model.state_dict(), path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # set random seed for all gpus


def binarize(T, nb_classes):
    device = T.device
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).to(device)
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


def get_config_from_cmd(key, default=None, key_type=None, kvdict=None, convert_to_list=False):
    """
    It will return default value or value in the argument commend
    if default is None, key_type should be given
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
    return value


def update_configs_from_cmd(config_dict, prefix=''):
    """
    config_dict: {key: value} | object (__dict__ will be used)
    prefix: str, it's used to concat the key from arguments, for de-duplicating purpose. e.g., the original key is 'lr'
            and the prefix is 'train-', the key used in args should be 'train-lr'.

    """
    if isinstance(config_dict, object):
        config_dict = config_dict.__dict__

    for key, value in config_dict.items():
        value1 = get_config_from_cmd('{}{}'.format(prefix, key), default=value)
        config_dict[key] = value1


class StringTextfileSaver():
    def __init__(self, filepath):
        self.filepath = filepath

    def save(self, obj):
        with open(self.filepath, 'w') as f:
            f.write(str(obj))

class ModelSaver():
    def __init__(self, filepath):
        self.filepath = filepath

    def save(self, model):
        torch.save(model.state_dict(), self.filepath)


class ModelLoader():
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self, model):
        model.load_state_dict(torch.load(self.filepath))


class TimerContext(object):
    def __init__(self, print=True, name='<not named>'):
        self.start = 0
        self.end = 0
        self.print_time_use = print
        self.timer_name = name

    @property
    def timespan(self):
        return str(datetime.timedelta(seconds=self.end-self.start))

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        if self.print_time_use:
            print('time used by [{}] [{}]'.format(self.timer_name, self.timespan))