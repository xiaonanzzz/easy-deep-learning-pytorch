import os.path
import pandas as pd
import PIL
from pathlib import Path
from torchvision.datasets.folder import default_loader
import numpy as np

class ImageDataset:

    # names for image schema
    Image = 'image'     # image tensor
    Label = 'label'
    Index = 'index'
    LabelCode = 'label_code'
    Name = 'name'       # base name of image

    def __init__(self, im_paths, labels, transform=None, item_schema=('image', 'label_code'), verbose=False):
        """
        item schema: a tuple/array indicating what datasets should be returned,
        such as, image | label | index | label_code | name
        label_code starts from 0
        """
        self.im_paths = list(im_paths)
        self.transform = transform
        self.labels = list(labels)
        assert len(self.im_paths) == len(self.labels)
        self.label_set = set(labels)
        self.label_map = {l:i for i, l, in enumerate(self.label_set)}
        self.item_schema = item_schema

        if verbose:
            print('im_paths ==>', self.im_paths[:2], self.im_paths[-2:])
            print('labels ==>', self.labels[:2], self.labels[-2:])
            print('label set ==>', len(self.label_set))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ret = []
        for it in self.item_schema:
            if it == 'image':
                im = default_loader(self.im_paths[index])
                if self.transform is not None:
                    im = self.transform(im)
                ret.append(im)
            if it == 'label':
                ret.append(self.labels[index])
            if it == 'index':
                ret.append(index)
            if it == 'label_code':
                ret.append(self.label_map[self.labels[index]])
            if it == 'name':
                ret.append(os.path.basename(self.im_paths[index]))


        return tuple(ret)

    @property
    def num_labels(self) -> int:
        return len(self.label_set)

    def subset(self, I):
        impath = self.im_paths[I]
        labels = self.labels[I]
        return ImageDataset(impath, labels, transform=self.transform, item_schema=self.item_schema)


def convert_pandas_dataset_to_image_dataset(df: pd.DataFrame, image_root='', image_col='image', label_col='label', transform=None, encode_label=True) -> ImageDataset:
    
    labels = df[label_col].to_list()

    if len(image_root) > 0:
        rootd = Path(image_root).expanduser()
        assert rootd.exists()
        # add image root
        images = df[image_col].map(lambda x: (rootd / x)).to_list()
    else:
        images = df[image_col].to_list()
    item_schema = ('image', 'label_code') if encode_label else ('image', 'label')
    return ImageDataset(images, labels, transform=transform, item_schema=item_schema)



