from .batch_processing import *
import random
import numpy as np
from .experiments import *
import time
import datetime
import os

def prepare_path(path):
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)

def save_model(model, path):
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