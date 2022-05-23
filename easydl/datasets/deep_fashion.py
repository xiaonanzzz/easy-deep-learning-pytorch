"""

Deep fashion dataset: 
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

"""
from .image_dataset import ImageDataset
from pathlib import Path
import pandas as pd
from easydl.metrics import recall_in_k_pytorch, recall_in_k_query_index
import numpy as np
from easydl.config import get_config_from_cmd

def show_inshop_retrieval_dataset():
    """
    img.zip download link:
    https://drive.google.com/file/d/0B7EVK8r0v71pS2YxRE1QTFZzekU/view?usp=sharing&resourcekey=0-2n-dbZygJiZdm0UbEaFRTA

    partition file download link:
    https://drive.google.com/file/d/0B7EVK8r0v71pYVBqLXpRVjhHeWM/view?usp=sharing&resourcekey=0-rxJ2QcImN-IRo_Bv9QSXmg
    """
 
    source = get_config_from_cmd('source', default='~/data/deepfashion_inshop/list_eval_partition.txt')

    assert Path(source).exists()
    source = Path(source).expanduser().as_posix()

    # ignore the first row which is number of rows
    df = pd.read_table(source, header=1, delim_whitespace=True)
    print(df.shape, df.head(3))
    print(df.groupby('evaluation_status').count()['image_name'])


class DeepFashionInshopExperiment:
    """
    Organize the data folder in this way
    ~/data/deepfashion_inshop/ 
        img/                            -> all images unzip from img.zip 
        list_eval_partition.txt         -> data partition file
            3 columns:  image_name, item_id, evaluation_status
    """
    def __init__(self, data_root='~/data/deepfashion_inshop'):
        
        self.data_root = data_root
        self.k_list = [1, 10, 20, 30]
        self.train_classes = 3997

    def train_ds(self, image_transform=None):
        listp = Path(self.data_root).expanduser() / 'list_eval_partition.txt'
        image_root = Path(self.data_root).expanduser()
        
        df = pd.read_table(listp, header=1, delim_whitespace=True)
        df_sub = df[df.evaluation_status == 'train']
        image_arr = list(map(lambda x: (image_root / x).as_posix(), df_sub['image_name']))
        labels_arr = df_sub['item_id']
        print('unique classes', len(set(labels_arr)))
        return ImageDataset(image_arr, labels_arr, transform=image_transform, item_schema=('image', 'label_code'))

    def query_and_gallery(self, image_transform=None):
        listp = Path(self.data_root).expanduser() / 'list_eval_partition.txt'
        image_root = Path(self.data_root).expanduser()
        
        df = pd.read_table(listp, header=1, delim_whitespace=True)

        # get all labels
        df_query = df[df.evaluation_status == 'query']
        df_gallery = df[df.evaluation_status == 'gallery']

        label_set = list(np.unique(list(df_query.item_id) + list(df_gallery.item_id)))
        print('unique labels in query and gallery', len(label_set))

        query_image = list(map(lambda x: (image_root / x).as_posix(), df_query.image_name))
        query_label = list(map(lambda x: label_set.index(x), df_query.item_id))

        gallery_image = list(map(lambda x: (image_root / x).as_posix(), df_gallery.image_name))
        gallery_label = list(map(lambda x: label_set.index(x), df_gallery.item_id))

        query_ds = ImageDataset(query_image, query_label, transform=image_transform, item_schema=('image', 'label'))
        gallery_ds = ImageDataset(gallery_image, gallery_label, transform=image_transform, item_schema=('image', 'label'))
        return query_ds, gallery_ds


    def evaluate_model(self, model, image_transform=None):
        model.eval()
        query_ds, gallery_ds = self.query_and_gallery(image_transform=image_transform)

        recall_at_k = recall_in_k_query_index(model, query_ds, gallery_ds, self.k_list)
        print(recall_at_k)
        return recall_at_k


if __name__ == "__main__":
    from easydl.config import get_config_from_cmd

    func_name = get_config_from_cmd('action', 'show')
    if func_name == 'show':
        show_inshop_retrieval_dataset()

    if func_name == 'example':
        exp = DeepFashionInshopExperiment()
        train_ds = exp.train_ds()
        print(len(train_ds), train_ds[0])
        query_ds, gallery_ds = exp.query_and_gallery()
        print(len(query_ds), query_ds[0])
        print(len(gallery_ds), gallery_ds[0])



    


