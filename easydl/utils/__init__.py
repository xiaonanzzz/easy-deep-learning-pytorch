import torch
from .batch_processing import process_dataset_by_batch, concat_tensors_in_list_of_tuple_given_index
import random
import numpy as np

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


