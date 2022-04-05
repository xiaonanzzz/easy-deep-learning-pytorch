import glob
from pathlib import Path
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
import xml.etree.ElementTree as ET
from collections import namedtuple
from tqdm import tqdm
from torchvision.ops import box_convert
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter

"""
Examples: 

python -i -m easydl.datasets.logo_det_3k

# ===== create a yolo format dataset

# === mini version, 2 images in train, 1 image per class in test
prepare_for_yolo(True)

# full version
prepare_for_yolo(False)

"""



def getImagesInDir(rootdir):
    print('collecting all images under', rootdir)
    image_list = []

    return image_list


def convert(size, box):
    """
    size: (w, h)
    box : (xmin, xmax, ymin, ymax)
    """
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


class LogoDet3KConfig(object):
    def __init__(self, **kwargs):
        # original folder location
        self.original_dir = '~/data/LogoDet-3K'
        self.yolo_folder = '~/data/LogoDet-3K-yolo'
        self.original_annotation_name = 'original_annotation.csv'
        self.class_file_name = 'class_index.csv'
        self.yml_file_name = 'logo_3k.yml'
        self.test_ratio = 0.5
        self.random_seed = 7
        self.subset = False   # 2 images in train, 1 images in test
        self.__dict__.update(kwargs)


def read_annotation(in_file: [str, Path]):
    Annotation = namedtuple('LogoDet3KAnnotation',
                            ['class_name', 'difficult', 'image_w', 'image_h', 'xmin', 'xmax', 'ymin', 'ymax'])

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    annotation_arr = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))

        anno = Annotation(cls, difficult, w, h, b[0], b[1], b[2], b[3])
        annotation_arr.append(anno._asdict())
    return annotation_arr


def get_key_to_image_rel_path(cfg: LogoDet3KConfig):
    root_0 = Path(cfg.original_dir).expanduser().as_posix()
    images = glob.glob(root_0 + '/**/*.jpg', recursive=True)
    print('# images', len(images), images[0])

    key2img_path = {}  # path_key -> image relative path

    for img in images:
        x = Path(img).relative_to(root_0)
        key = Path(x).parent / Path(x).stem
        key2img_path[key] = x

    return key2img_path


def get_all_annotations(cfg: LogoDet3KConfig):
    root_0 = Path(cfg.original_dir).expanduser().as_posix()

    labels = glob.glob(root_0 + '/**/*.xml', recursive=True)
    print('# labels', len(labels), labels[0])

    key2label_path = {}  # path_key -> label file relative path
    all_anno_arr = []
    for labelp in tqdm(labels, desc='Processing label xml files'):
        x = Path(labelp).relative_to(root_0)
        key = Path(x).parent / Path(x).stem
        key2label_path[key] = x
        anno_arr = read_annotation(labelp)
        for a in anno_arr:
            a['key'] = key
        all_anno_arr.extend(anno_arr)

    annotation_df = pd.DataFrame(all_anno_arr)

    yolo_path = Path(cfg.yolo_folder).expanduser()
    yolo_path.mkdir(exist_ok=True)
    annotation_df.to_csv(yolo_path / cfg.original_annotation_name, index=False)

def generate_yolo_folder(cfg: LogoDet3KConfig):
    yolo_path = Path(cfg.yolo_folder).expanduser()
    annotation_df = pd.read_csv(yolo_path / cfg.original_annotation_name)
    class_arr = sorted(list(annotation_df['class_name'].unique()))

    # the actual number of classes is 2993
    print('# classes', len(class_arr))
    print(class_arr)

    # split train / test folder
    split_df = annotation_df[['class_name', 'key']].drop_duplicates('key')
    print('# images before split', split_df.shape)

    train_set, test_set = train_test_split(split_df['key'], test_size=0.3, random_state=cfg.random_seed, stratify=split_df['class_name'])
    print('train set', len(train_set), 'test set', len(test_set))
    annotation_df['is_test'] = annotation_df['key'].isin(test_set)

    pd.Series(class_arr, name='class_name').to_csv(Path(cfg.yolo_folder).expanduser() / cfg.class_file_name, index=False)

    train_counter, test_counter = Counter(), Counter()

    for key in tqdm(annotation_df['key'].unique()):
        df = annotation_df[annotation_df['key'] == key].copy()
        df['image_loc'] = 0
        if df.shape[0] == 0:
            continue
        class_name_0 = df.iloc[0].class_name
        if cfg.subset:
            if df.iloc[0]['is_test'] and test_counter[class_name_0] >= 1:
                continue
            if not df.iloc[0]['is_test'] and train_counter[class_name_0] >= 2:
                continue
            if df.iloc[0]['is_test']:
                test_counter.update([class_name_0])
            else:
                train_counter.update([class_name_0])

        src_image = Path(cfg.original_dir).expanduser() / (key + '.jpg')
        assert src_image.exists()

        # move image
        if df.iloc[0]['is_test']:
            out_image = Path(cfg.yolo_folder).expanduser() / 'images' / 'test' / (key.replace('/', '_') + '.jpg')
        else:
            out_image = Path(cfg.yolo_folder).expanduser() / 'images' / 'train' / (key.replace('/', '_') + '.jpg')

        out_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_image, out_image)

        # generate label text file
        out_label = str(out_image).replace('images', 'labels')
        out_label = out_label.replace('.jpg', '.txt')
        Path(out_label).parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_label, 'w')
        for row in df.itertuples():
            bbox = convert((row.image_w, row.image_h), (row.xmin, row.xmax, row.ymin, row.ymax))

            out_file.write(str(class_arr.index(row.class_name)) + " " + " ".join([str(a) for a in bbox]) + '\n')
        out_file.close()


def prepare_for_yolo(mini=False):
    cfg = LogoDet3KConfig()
    if mini:
        cfg.yolo_folder = '~/data/LogoDet-3K-yolo-mini'
        cfg.subset = True

    if (Path(cfg.yolo_folder).expanduser() / cfg.original_annotation_name).exists():
        print('original annotation already generated...')
    else:
        get_all_annotations(cfg)

    generate_yolo_folder(cfg)

def prepare_for_yolo_yml(mini=False):
    cfg = LogoDet3KConfig()
    if mini:
        cfg.yolo_folder = '~/data/LogoDet-3K-yolo-mini'
        cfg.subset = True

    # generate yml
    root = Path(cfg.yolo_folder)
    # load
    class_arr = pd.read_csv(root.expanduser() / cfg.class_file_name)['class_name'].tolist()

    with open(root.expanduser() / cfg.yml_file_name, 'w') as f:
        f.write('path: {}\n'.format(root.as_posix()))
        f.write('train: {}\n'.format('images/train'))
        f.write('val: {}\n'.format('images/test'))
        f.write('nc: {}\n'.format(len(class_arr)))
        f.write('names: {}\n'.format(class_arr))

