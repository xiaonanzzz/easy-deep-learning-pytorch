import torch
from tqdm import tqdm

def process_dataset_by_batch(dataset, model, input_index=0, batch_size=32, save_index=1,
                             device=torch.device('cpu'), out_device=torch.device('cpu')):
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True,
    )
    out = []
    save = []
    for data_batch in tqdm(dl):
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

    if isinstance(input_index, int) or isinstance(input_index, list):
        return out, save
    return out

def batch_process_x_y_dataset_and_concat(dataset, model, **kwargs):
    x, y = process_dataset_by_batch(dataset, model, **kwargs)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    return x, y

def concat_tensors_in_list_of_tuple_given_index(tensor_list_tuple, index_in_tuple, concat_dim=0):
    list_of_tensor = []
    for one in tensor_list_tuple:
        list_of_tensor.append(one[index_in_tuple])

    return torch.cat(list_of_tensor, dim=concat_dim)
