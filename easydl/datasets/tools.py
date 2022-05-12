from tkinter import W
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import shutil

VocAnnotationKVType = {
    'width': int,
    'height': int, 
    'name': str,
    'xmin': int,
    'xmax': int,
    'ymin': int,
    'ymax': int, 
}

YoloAnnotationKVType = {
    'class': int,   # class index
    'xc': float,    
    'yc': float,
    'bw': float,    # box width
    'bh': float,    # box height
}


def read_voc_annotation(in_file) -> list:
    """ return a list of dict
    {'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'}
    image width, image height, class name/id, bbox coordinates
    
    """
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')

    annotation_arr = []
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        anno = {
            'width': int(size.find('width').text), 
            'height': int(size.find('height').text), 
            'name': obj.find('name').text,
        }
        for key in ['xmin', 'xmax', 'ymin', 'ymax']:
            anno[key] = int(xmlbox.find(key).text)

        annotation_arr.append(anno)
    return annotation_arr


def voc2yolo(voc: dict, class_arr):
    w, h = voc['width'], voc['height']
    xmin, xmax, ymin, ymax = voc['xmin'], voc['xmax'], voc['ymin'], voc['ymax']

    xc = (xmin + xmax) / (2.0 * w)
    yc = (ymin + ymax) / (2.0 * h)
    bw = (xmax - xmin) / w
    bh = (xmax - xmin) / h
    return {
        'class': class_arr.index(voc['name']),
        'xc': xc,
        'yc': yc,
        'bw': bw,
        'bh': bh,
    }

def write_yolo_labels(annotation_arr, ofile):
    with open(ofile, 'wt') as f:
        for anno in annotation_arr:

            f.write(' '.join(map(lambda x: str(x), [anno['class'], anno['xc'], anno['yc'], anno['bw'], anno['bh']])))
            f.write('\n')



def voc_dataset_to_yolo(image_df, yolo_image_dir, yolo_label_dir, class_arr):
    """
    image_df: a pandas df, 2 columns, image -> image path, annotation -> annotation path
    yolo_image_dir: the dir for paste image to
    yolo_label_dir: the dir for label files
    class_arr: 
    """

    assert 'image' in image_df
    assert 'annotation' in image_df

    os.makedirs(yolo_image_dir, exist_ok=True)
    os.makedirs(yolo_label_dir, exist_ok=True)

    for (imagef, annof) in image_df[['image', 'annotation']].itertuples(index=False):
        base_name = os.path.basename(imagef)
        shutil.copy(imagef, os.path.join(yolo_image_dir, base_name))

        # 
        voc_annos = read_voc_annotation(annof)
        yolo_annos = [voc2yolo(x, class_arr) for x in voc_annos]

        fname = Path(base_name).with_suffix('.txt')

        write_yolo_labels(yolo_annos, os.path.join(yolo_label_dir, fname))

