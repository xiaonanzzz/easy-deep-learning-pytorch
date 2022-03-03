import datetime
import os
import random
import time

import torch
import numpy as np


def all_to_device(*args, device=None):
    o = [a.to(device) for a in args]
    return tuple(o)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.data.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def expand_path(path):
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return path


def prepare_path(path):
    path = expand_path(path)
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)


def save_model(model, path):
    """ create the dirs for the model path, then save the model"""
    prepare_path(path)
    torch.save(model.state_dict(), path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # set random seed for all gpus


def binarize(T, nb_classes):
    device = T.device
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).to(device)
    return T


""" =================================== timer helper functions ============================== """
_timers = {}

def start_timer(name):
    _timers[name] = time.time()


def end_timer(name):
    # always return float, if not a registered timer, return 0
    end_t = time.time()
    tspan = end_t - _timers.get(name, end_t)
    if name in _timers:
        del _timers
    return tspan


def profile_parameters(params):
    for param in params:
        yield {'size': param.size(), 'require_grad': param.data.requires_grad}
