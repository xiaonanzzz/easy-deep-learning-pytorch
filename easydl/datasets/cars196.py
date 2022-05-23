from .image_dataset import ImageDataset, convert_pandas_dataset_to_image_dataset
from pathlib import Path
import pandas as pd
from easydl.metrics import recall_in_k_pytorch, recall_in_k_query_index, recall_in_k_self_retrieval
import numpy as np
from easydl.config import get_config_from_cmd
import scipy.io

"""

assume that the root dir is
~/data/cars196
    car_ims/            # 16185 images
    cars_annos.mat

# download images
wget http://ai.stanford.edu/~jkrause/car196/car_ims.tgz

# download annotation file
wget http://ai.stanford.edu/~jkrause/car196/cars_annos.mat


train split


"""


def show_dataset():
    data_root = get_config_from_cmd('source', default='~/data/cars196')

    labelf = Path(data_root).expanduser() / 'cars_annos.mat'

    cars = scipy.io.loadmat(labelf) 
    # cars.keys() -> dict_keys(['__header__', '__version__', '__globals__', 'annotations', 'class_names'])
    print(cars.keys())
    print('annotations', type(cars['annotations']), cars['annotations'].shape)
    print('class_names', type(cars['class_names']), cars['class_names'].shape)

    print('annotations[0]', type(cars['annotations'][0]), cars['annotations'][0, 0])

    # each a has 6 items -> image, bbox (4 items), label (start from 1)
    ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
    im_paths = [a[0][0] for a in cars['annotations'][0]]

    class_names = list(x[0] for x in cars['class_names'][0])

    df = pd.DataFrame()
    df['image'] = im_paths
    df['label'] = ys
    df['split'] = df['label'].map(lambda x: 'train' if x <98 else 'test')
    df['label_name'] = df['label'].map(lambda x: class_names[x])

    

    df.to_csv(labelf.with_name('cars196_split.csv'), index=False)


def get_data_split(data_root):
    """
    image,label,split,label_name
    car_ims/000001.jpg,0,train,AM General Hummer SUV 2000
    car_ims/000002.jpg,0,train,AM General Hummer SUV 2000
    car_ims/000003.jpg,0,train,AM General Hummer SUV 2000
    car_ims/000004.jpg,0,train,AM General Hummer SUV 2000
    """
    labelf = Path(data_root).expanduser() / 'cars_annos.mat'

    cars = scipy.io.loadmat(labelf) 
    ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
    im_paths = [a[0][0] for a in cars['annotations'][0]]

    class_names = list(x[0] for x in cars['class_names'][0])

    df = pd.DataFrame()
    df['image'] = im_paths
    df['label'] = ys
    df['split'] = df['label'].map(lambda x: 'train' if x <98 else 'test')
    df['label_name'] = df['label'].map(lambda x: class_names[x])
    return df


class Cars196MetricLearningExp:
    def __init__(self, data_root='~/data/cars196'):
        self.data_root = data_root
        self.k_list = [1, 2, 4, 8]
        self.train_classes = 98
        self.df = get_data_split(data_root)

    def train_ds(self, image_transform=None):
        df = self.df
        df = df[df.split == 'train'].reset_index()
        image_root = Path(self.data_root).expanduser().as_posix()
        ds = convert_pandas_dataset_to_image_dataset(df, image_root=image_root, transform=image_transform, encode_label=True)
        return ds

    def test_ds(self, image_transform=None):
        df = self.df
        df = df[df.split == 'test'].reset_index()
        image_root = Path(self.data_root).expanduser().as_posix()
        ds = convert_pandas_dataset_to_image_dataset(df, image_root=image_root, transform=image_transform, encode_label=True)
        return ds

    def evaluate_model(self, model, image_transform=None):
        model.eval()
        test_ds = self.test_ds(image_transform=image_transform)
        recall_at_k = recall_in_k_self_retrieval(model, test_ds, self.k_list)
        print(recall_at_k)
        return recall_at_k

if __name__ == "__main__":

    func_name = get_config_from_cmd('action', 'show')
    if func_name == 'show':
        show_dataset()
    if func_name == 'exp':
        exp = Cars196MetricLearningExp()
        train_ds = exp.train_ds()
        print(len(train_ds), train_ds[0])
        test_ds = exp.test_ds()
        print(len(test_ds), test_ds[0])


