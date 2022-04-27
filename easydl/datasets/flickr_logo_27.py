
import glob
from pathlib import Path
import posixpath
import pandas as pd
import xml.etree.ElementTree as ET
from collections import namedtuple
from tqdm import tqdm
import shutil
from collections import Counter
from PIL import Image
import numpy as np
from torchvision.io import read_image

"""
This 

Examples: 

python -i -m easydl.datasets.logo_det_3k

# ===== create a yolo format dataset

# === mini version, 2 images in train, 1 image per class in test
prepare_for_yolo(True)

# full version
prepare_for_yolo(False)

"""


class FlickrLogo27Helper(object):
    def __init__(self, **kwargs):
        # original folder location
        self.source = '~/data/flickr_logos_27_dataset'
        self.dest = '~/data/flickr_logos_27_dataset_yolo'
        self.annotation_fname = 'flickr_logos_27_dataset_training_set_annotation.txt'
        self.source_image_folder_name = 'flickr_logos_27_dataset_images'

        self.original_annotation_name = 'original_annotation.csv'
        self.class_file_name = 'class_index.csv'
        self.yml_file_name = 'flickr_logos_27.yml'
        self.verbose = 0    # verbose level 0, 1, 2, ...        0 should be lowest level, print least amount of info
        self.__dict__.update(kwargs)

    def parser(self, ):
        import argparse
        pa = argparse.ArgumentParser()
        pa.add_argument('--source', default=self.source)
        pa.add_argument('--dest', default=self.dest)
        pa.add_argument('--verbose', default=self.verbose, help='verbose level, 0 -> 2, 0 shows least info')

        return pa
    
    def parse(self, update_value=True):
        pa = self.parser()
        opt = pa.parse_known_args()
        opt = opt[0]

        if update_value:
            self.__dict__.update(opt.__dict__)
        return opt

    def read_annotation(self):
        df = pd.read_csv(Path(self.source).expanduser() / self.annotation_fname, sep='\s', header=None, 
        names=['image', 'label', 'subset', 'x1', 'y1', 'x2', 'y2'])

        return df

    
    def show_annotation_stat(self):
        df = self.read_annotation()
        
        print('unique values')
        print(df.nunique())

        df0 = df.groupby('subset').count()['image']
        print(df0)


    def generate_yolo_folder(self):
        """
        Yolo label file format (sep by)

        """
        src_img_dir = Path(self.source).expanduser() / self.source_image_folder_name
        yolo_dir = Path(self.dest).expanduser()

        imgdir = yolo_dir / 'images' / 'train'
        labeldir = yolo_dir / 'labels' / 'train'
        imgdir.mkdir(parents=True, exist_ok=True)
        labeldir.mkdir(parents=True, exist_ok=True)

        alldf = self.read_annotation()

        class_arr = sorted(list(alldf['label'].unique()))

        box_count = 0

        for image in alldf['image'].unique():
            image_f: Path = imgdir / image        # dest image file
            src_f: Path = src_img_dir / image     # source image file
            label_f = labeldir / image_f.with_suffix('.txt').name   # change file name to '.txt'
            
            shutil.copy(src_f, image_f)

            onedf = alldf[alldf.image == image]
            onedf = onedf.drop_duplicates(['x1', 'x2', 'y1', 'y2'])
            
            # read image size
            imgx = np.array(Image.open(src_f))
            
            h, w, _ = imgx.shape # height, width, channels

            xc = ((onedf.x1 + onedf.x2)/2.0) / w
            yc = ((onedf.y1 + onedf.y2)/2.0 - 1) / h
            bw = (onedf.x2 - onedf.x1) / w
            bh = (onedf.y2 - onedf.y1) / h

            label_id = onedf.label.map(lambda x: class_arr.index(x))

            with open(label_f, 'w') as f:
                for arr in zip(label_id, xc, yc, bw, bh):
                    f.write(' '.join(map(str, arr)))
                    f.write('\n')

            if self.verbose > 0:
                print('image:', image, 'shape', imgx.shape, '#boxes', len(onedf), )
            if self.verbose >= 2:
                print(onedf)

            box_count += len(label_id)

        with open(yolo_dir.expanduser() / self.yml_file_name, 'w') as f:
            f.write('path: {}\n'.format(yolo_dir.as_posix()))
            f.write('train: {}\n'.format('images/train'))
            f.write('nc: {}\n'.format(len(class_arr)))
            f.write('names: {}\n'.format(class_arr))
            f.write('box_count: {}\n'.format(box_count))
            f.write('images: {}\n'.format(alldf.image.nunique()))


def main():
    helper = FlickrLogo27Helper()
    print(helper.parse())

    # df = helper.read_annotation()
    # print(df)
    helper.show_annotation_stat()

    helper.generate_yolo_folder()


if __name__ == '__main__':
    main()
