import glob
from pathlib import Path
import pandas as pd
from requests import options
import torch
from torchvision.datasets import ImageFolder
import xml.etree.ElementTree as ET
from collections import namedtuple
from tqdm import tqdm
from torchvision.ops import box_convert
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter


class FoodLogoDatasetHelper:
    def __init__(self, data_root='~/data/FoodLogoDet-1500') -> None:
        self.data_root = data_root
    

    def generate_image_list(self):
        rootp = Path(self.data_root).expanduser()

        traindf = pd.read_csv(rootp / 'ImageSets/Main/train.txt', header=None)
        print('traindf', traindf.shape)

        valdf = pd.read_csv(rootp / 'ImageSets/Main/val.txt', header=None)
        print('valdf', valdf.shape)

        testdf = pd.read_csv(rootp / 'ImageSets/Main/test.txt', header=None)
        print('testdf', testdf.shape)

        traindf['split'] = 'train'
        valdf['split'] = 'val'
        testdf['split'] = 'test'

        df = pd.concat( [traindf, valdf, testdf] )
        df.rename(columns={0: 'id'}, inplace=True)
        df['image'] = df['id'].astype(int).map(lambda x: 'JPEGImages/{:06}.jpg'.format(x))
        df['annotation'] = df['id'].astype(int).map(lambda x: 'Annotations/{:06}.xml'.format(x))
        df.to_csv(rootp / 'image_list.csv', index=False)

        print('all images: ', df.shape, 'head rows: \n', df.head(3))

        try:
            print('checking existance of images, control+c to stop...')
            for img in tqdm(df['image']):
                _p = rootp / img
                assert _p.exists(), _p
            print('check completed! ')
            print('checking existance of annotations, control+c to stop...')
            for anno in tqdm(df['annotation']):
                _p = rootp / anno
                assert _p.exists(), _p
            print('check completed! ')
        except AssertionError as e:
            print('checking failed on', e)
        except KeyboardInterrupt as e:
            print('stop checking...')
            pass

        return df


    def yolo(self):
        from easydl.datasets.tools import voc_dataset_to_yolo
        # convert data to yolo format
        """
        output dir: 
        yolo/
            images/
                train/  # 70362
                val/    # 9918
                test/   # 19488
            labels/
                train/
                val/
                test/
        """
        rootp = Path(self.data_root).expanduser()
        metap = rootp / 'image_list.csv'
        classp = rootp / 'class.txt'
        yolo_dir = rootp / 'yolo'
        yolo_image = rootp / 'yolo/images'
        yolo_label = rootp / 'yolo/labels'

        yolo_image.mkdir(parents=True, exist_ok=True)
        yolo_label.mkdir(parents=True, exist_ok=True)
        
        # read class list
        with open(classp) as f:
            buffer = f.read()
            class_arr = eval("[" + buffer + "]")
            print(len(class_arr), class_arr[0])

        imagedf = pd.read_csv(metap)
        print('imagedf', imagedf.shape, '\n', imagedf.head(3))


        for split in ['train', 'val', 'test']:
            subdf = imagedf.query(f'split == "{split}"').copy()
            print('split', split, 'shape', subdf.shape, '\n', subdf.head(3))
            subdf['image'] = subdf['image'].map(lambda x: str(rootp / x))
            subdf['annotation'] = subdf['annotation'].map(lambda x: str(rootp / x))

            voc_dataset_to_yolo(subdf, yolo_image / split, yolo_label / split, class_arr)
            
        with open(yolo_dir.expanduser() / 'food-logo.yml', 'w') as f:
            f.write('path: {}\n'.format(yolo_dir.as_posix()))
            f.write('train: {}\n'.format('images/train'))
            f.write('val: {}\n'.format('images/val'))
            f.write('test: {}\n'.format('images/test'))
            f.write('nc: {}\n'.format(len(class_arr)))
            f.write('names: {}\n'.format(class_arr))

    
    def main(self):
        import argparse
        pa = argparse.ArgumentParser()
        pa.add_argument('--action', default='generate')
        
        args = pa.parse_args()
        self.args = args

        if args.action == 'generate':
            self.generate_image_list()
        elif args.action == 'yolo':
            self.yolo()
        else:
            raise NotImplementedError(f'unknown action')
        
    
if __name__ == '__main__':
    helper = FoodLogoDatasetHelper()
    helper.main()

