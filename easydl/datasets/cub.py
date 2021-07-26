import torchvision
import os
import pandas as pd
from easydl.datasets import ImageLoader
import numpy as np
from torchvision.transforms import ToTensor, Resize, Normalize


_default_image_transformer = torchvision.transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(0.45, 0.22),      # simple version from https://pytorch.org/vision/stable/models.html
])

class CUBirdsHelper(object):
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
        self.meta_df = meta_df


class Cub2011MetricLearningDS(CUBirdsHelper, ImageLoader):

    def __init__(self, root, *args, split='train', image_transform=_default_image_transformer, **kwargs):
        super(Cub2011MetricLearningDS, self).__init__(root, *args, image_transform=image_transform, **kwargs)
        self.split = split

        if self.split == 'train':
            self.data = self.meta_df[self.meta_df['label'].isin(np.arange(1, 100 + 1))]
        elif self.split == 'test':
            self.data = self.meta_df[self.meta_df['label'].isin(np.arange(100+1, 200 + 1))]
        else:
            raise ValueError('wrong split mode, only accept train/test')
        self.data.reset_index()
        print('cub 2011 metric learning dataset data size', self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.image_folder, sample['image_path'])
        target = sample['label'] - 1

        return self.load_image(path), target

if __name__ == '__main__':
    ds = Cub2011MetricLearningDS('/Users/xiaonzha/data/CUB_200_2011', split='test')
    print(ds[0])