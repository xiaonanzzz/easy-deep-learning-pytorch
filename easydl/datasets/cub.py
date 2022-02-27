import torchvision
import os
import pandas as pd
from easydl.datasets.image_dataset import ImageDataset
import numpy as np
from easydl.image_transform import resnet_transform_train, resnet_transform_test, make_transform_train_v1, make_transform_test_v1
from easydl.metrics import recall_in_k_self_retrieval
from easydl.config import get_config_from_cmd

_metric_learning_evaluation_k_list = [1, 2, 4, 8]


class CUBirdsHelper(object):
    """
    meta_df schema and example
    image_id                                                           1
    image_path         001.Black_footed_Albatross/Black_Footed_Albatr...
    label                                                              1
    is_training_img                                                    0

    meta_df has 11788 rows, and 4 columns

    """
    def __init__(self, root, *args, **kwargs):
        super(CUBirdsHelper, self).__init__(*args, **kwargs)
        self.root = root
        self.image_folder = os.path.join(self.root, 'CUB_200_2011', 'images')
        image_list = os.path.join(self.root, 'CUB_200_2011', 'images.txt')
        meta_df = pd.read_csv(image_list, sep=' ', names=['image_id', 'image_path'], header=None)

        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['image_id', 'label'])
        meta_df = meta_df.merge(image_class_labels, on='image_id')
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['image_id', 'is_training_img'])
        meta_df = meta_df.merge(train_test_split, on='image_id')
        # meta df is merged datasets profile
        self.meta_df = meta_df


class CubClassificationDS:
    """
    5,994 training images and 5,794 testing data
    """
    def __init__(self, root, split='train', image_transform=None, item_schema=('image', 'label_code'), **kwargs):
        self.cub = CUBirdsHelper(root)

        meta_df = self.cub.meta_df
        self.split = split
        if self.split == 'train':
            self.data = meta_df[meta_df['is_training_img'] == 1]
        elif self.split == 'test':
            self.data = meta_df[meta_df['is_training_img'] == 0]
        else:
            raise ValueError('wrong split mode, only accept train/test')
        self.data = self.data.reset_index(drop=True)

        image_path = self.data.image_path.map(lambda x: os.path.join(self.cub.image_folder, x))
        labels = self.data.label

        self.dataset = ImageDataset(image_path, labels, transform=image_transform, item_schema=item_schema)

        print('cub 2011 classification data, split:', split, 'shape:', self.data.shape)

    def show_profile(self):
        print('unique values per column\n', self.data.nunique())
        print('head ', self.data.head())

    def change_image_transform(self, image_transform):
        self.dataset.transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class Cub2011MetricLearningDS:
    """
    train datasets has 5864, label in [1, 100]
    test datasets has 5924, label in [101, 200]
    """
    def __init__(self, root, split='train', image_transform=None, item_schema=('image', 'label_code'), **kwargs):
        self.cub = CUBirdsHelper(root)

        meta_df = self.cub.meta_df
        self.split = split
        if self.split == 'train':
            self.data = meta_df[meta_df['label'].isin(np.arange(1, 100 + 1))]
        elif self.split == 'test':
            self.data = meta_df[meta_df['label'].isin(np.arange(100+1, 200 + 1))]
        else:
            raise ValueError('wrong split mode, only accept train/test')
        self.data = self.data.reset_index()

        image_path = self.data.image_path.map(lambda x: os.path.join(self.cub.image_folder, x))
        labels = self.data.label

        self.dataset = ImageDataset(image_path, labels, transform=image_transform, item_schema=item_schema)

        print('cub 2011 metric learning dataset datasets size', split, self.data.shape)

    def change_image_transform(self, image_transform):
        self.dataset.transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CubClassificationExperiment:
    def __init__(self, image_size):
        self.data_path = get_config_from_cmd('data_path', '~/data/CUB_200_2011')
        self.train_ds = CubClassificationDS(self.data_path, split='train', image_transform=make_transform_train_v1(image_size=image_size))
        self.test_ds = CubClassificationDS(self.data_path, split='test', image_transform=make_transform_test_v1(image_size=image_size))
        self.n_classes = 200


class CubMetricLearningExperiment:
    def __init__(self):
        self.data_path = get_config_from_cmd('data_path', '~/data/CUB_200_2011')
        self.train_ds = Cub2011MetricLearningDS(self.data_path, image_transform=resnet_transform_train)
        self.test_ds = Cub2011MetricLearningDS(self.data_path, split='test', image_transform=resnet_transform_test)
        self.k_list = _metric_learning_evaluation_k_list
        self.train_classes = 100

    def evaluate_model(self, model):
        model.eval()
        recall_at_k = recall_in_k_self_retrieval(model, self.test_ds, self.k_list)
        print(recall_at_k)
        return recall_at_k

