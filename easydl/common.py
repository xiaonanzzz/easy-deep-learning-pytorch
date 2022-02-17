import datetime
import os
import random
import time

import torch
import numpy as np


class TrainAccuracyAverage():
    def __init__(self):
        self.prediction = []
        self.truth = []
        self.scores = []

    def update(self, pred, truth):
        """
        pred: torch.Tensor N*M, N*1, N,
        truth: torch.Tensor  N
        """
        truth = to_numpy(truth)
        assert truth.ndim == 1
        pred = np.squeeze(to_numpy(pred))
        assert pred.ndim == 2 or pred.ndim == 1
        self.truth.extend(truth)
        if pred.ndim == 2:
            pred = pred.argmax(axis=1)
        self.prediction.extend(pred)

    def accuracy(self):
        pred = np.array(self.prediction)
        truth = np.array(self.truth)
        return (pred == truth).mean()


class ModelSaveEpochHook:
    def __init__(self, model, filepath):
        self.model = model
        self.filepath = filepath

    def __call__(self, *args, **kwargs):
        save_model(self.model, self.filepath)


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