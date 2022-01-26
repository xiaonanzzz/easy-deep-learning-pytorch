import torchvision
import os
import pandas as pd
from easydl.datasets import ImageDataset
import numpy as np
from torchvision.transforms import ToTensor, Resize, Normalize


_default_image_transformer = torchvision.transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(0.45, 0.22),      # simple version from https://pytorch.org/vision/stable/models.html
])


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
        # meta df is merged data profile
        self.meta_df = meta_df


class Cub2011MetricLearningDS:
    """
    train data has 5864, label in [1, 100]
    test data has 5924, label in [101, 200]
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

        print('cub 2011 metric learning dataset data size', split, self.data.shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

if __name__ == '__main__':
    from easydl.utils import expand_path
    cub = CUBirdsHelper(expand_path('~/data/CUB_200_2011'))
    print(cub.meta_df)
    trainds = Cub2011MetricLearningDS(expand_path('~/data/CUB_200_2011'), item_schema=('image', 'label_code', 'name'))
    print(trainds.data.shape)
    print(trainds.data.nunique())
    print(trainds[0])
    testds = Cub2011MetricLearningDS(expand_path('~/data/CUB_200_2011'), split='test', item_schema=('image', 'label_code', 'name'))
    print(testds.data.shape)
    print(testds.data.nunique())
    print(testds[0])





