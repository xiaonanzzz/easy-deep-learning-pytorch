import argparse
import json

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision.io import read_image, write_png
from torchvision.io.image import ImageReadMode
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import torch
from collections import Counter

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('input_image')
    pa.add_argument('--output_prefix', '-o')
    args = pa.parse_args()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    img = read_image(args.input_image, ImageReadMode.RGB)
    print('image shape', img.shape, img.min(), img.max())

    x = F.convert_image_dtype(img, dtype=torch.float)
    x = torch.stack([x])

    y = model(x)[0] # take only one from list
    print('prediction keys', y.keys())

    boxes = y['boxes']
    result = draw_bounding_boxes(img, boxes, width=3)
    opath = (args.output_prefix or args.input_image) + '-faster-rcnn-detection-result.png'
    write_png(result, opath)

    prediction_data = {}
    prediction_data['number of boxes'] = len(y['boxes'])
    prediction_data['label distribution'] = Counter(y['labels'].detach().numpy().tolist())
    prediction_data['detections'] = [{'box': y['boxes'][i].detach().numpy().tolist(),
                                      'label': y['labels'][i].item(),
                                      'score': float(y['scores'][i].item())}
                                     for i in range(len(y['boxes']))]
    print(prediction_data)
    with open((args.output_prefix or args.input_image) + '-faster-rcnn-detection-result.json', 'wt') as f:
        json.dump(prediction_data, f, indent=1)


if __name__ == '__main__':
    main()