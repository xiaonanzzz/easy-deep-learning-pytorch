import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import numpy as np

def process_dataset_by_batch(dataset, model, input_index=0, batch_size=32, save_index=1, out_device=torch.device('cpu'),
                             tqdm_disable=True,
                             tqdm_description=''):
    """

    :param dataset:         a pytorch dataset class, e.g. ds[index] -> x0, x1, x2, ...
    :param model:           a pytorch model, which can transform x into a tensor/score
    :param input_index:     which one is the input value of model
    :param batch_size:
    :param save_index:      which index/indices should be kept and return
    :param out_device:      which device to save the result
    :return:
    """
    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True,
    )
    out = []
    save = []
    for data_batch in tqdm(dl, disable=tqdm_disable, desc=tqdm_description):
        with torch.autograd.no_grad():
            x = data_batch[input_index]
            x = x.to(device)
            y = model(x)
            y = y.to(out_device)
            out.append(y)
            if isinstance(save_index, int):
                save.append(data_batch[save_index])
            elif isinstance(save_index, list):
                save.append([data_batch[idx] for idx in save_index])

    if isinstance(save_index, int) or isinstance(save_index, list):
        return out, save
    return out

def batch_process_x_y_dataset_and_concat(dataset, model, **kwargs):
    """ Given a dataset consisting of input(x) and target(y),
    run the given <model> on <x> -> <x'>
    return x' and y
    For example, given a dataset of images and labels, a model of embedding model, the output are embeddings and target
    """
    x, y = process_dataset_by_batch(dataset, model, **kwargs)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    return x, y

def concat_tensors_in_list_of_tuple_given_index(tensor_list_tuple, index_in_tuple, concat_dim=0):
    list_of_tensor = []
    for one in tensor_list_tuple:
        list_of_tensor.append(one[index_in_tuple])

    return torch.cat(list_of_tensor, dim=concat_dim)


def batch_process_tensor(x, model, **kwargs):
    """ parameters see process_dataset_by_batch """
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.FloatTensor(x)

    ds = TensorDataset(x)
    x = process_dataset_by_batch(ds, model, save_index=None, **kwargs)
    x = torch.cat(x, dim=0)
    if is_numpy:
        return x.numpy()
    return x

