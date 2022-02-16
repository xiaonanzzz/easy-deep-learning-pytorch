import os.path

import PIL
from torchvision.datasets.folder import default_loader
import numpy as np

class ImageDataset:
    def __init__(self, im_paths, labels, transform=None, item_schema=('image', 'label_code')):
        """
        item schema: a tuple/array indicating what datasets should be returned,
        such as, image | label | index | label_code | name
        label_code starts from 0
        """
        self.im_paths = list(im_paths)
        self.transform = transform
        self.labels = list(labels)
        assert len(self.im_paths) == len(self.labels)
        self.label_set = np.unique(labels)
        self.label_map = {l:i for i, l, in enumerate(self.label_set)}
        self.item_schema = item_schema

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

    def subset(self, I):
        impath = self.im_paths[I]
        labels = self.labels[I]
        return ImageDataset(impath, labels, transform=self.transform, item_schema=self.item_schema)

