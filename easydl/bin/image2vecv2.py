import argparse
import os
import sys
from easydl.models.convnext import convnext_base
from easydl.models.mlp_model import L2Normalization
import pandas as pd
from datetime import datetime
from easydl.image_transform import resnet_transform_test
import json
import torch
from PIL import Image
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
    
class ImageDataset:
    def __init__(self, images, transform=None):
        self.images = list(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_fp = self.images[index]

        try:
            im = Image.open(image_fp).convert('RGB')
            valid = 1
        except:
            # create a black image
            im = Image.new(mode='RGB', size=(300, 300), color=0)
            valid = 0
        if self.transform is not None:
            im = self.transform(im)

        return (im, index, valid)
    

def main():
    parser = argparse.ArgumentParser(description="Transform images into vectors")
    parser.add_argument("--input_csv", '-c', help="Image to be predicted", type=str, required=True)
    parser.add_argument("--model", help="confidence threshold", default='convnext-base', type=str)
    parser.add_argument('--image_root', default='', type=str, help='the root dir of image path')
    parser.add_argument('--batch_size', type=int, default=200, help='the batch size for splitting the original data')
    parser.add_argument('--device', default='auto')
    parser.add_argument('-o', '--out_dir', type=str, default='tmp/image2vec-{}'.format(str(datetime.today()).replace(' ', '-')))
    args = parser.parse_args()

    # prepare run 
    batch_size = args.batch_size
    df = pd.read_csv(args.input_csv)
    # concat the iamge root with image path provided by csv
    images = df['path'].map(lambda x: os.path.join(args.image_root, x))

    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('')
    logger.addHandler(logging.FileHandler(odir / 'image2vec.log'))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)


    if args.model == 'convnext-base':
        model = convnext_base(pretrained=True, in_22k=True, num_classes=21841)
        model.head = L2Normalization()
        dataset = ImageDataset(images, resnet_transform_test)
    else: 
        raise NotImplemented
    
    logger.info(f'total images: {len(images)}')


    # run model on all dataset
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    # device = torch.device('cuda')
    logger.info(f'cuda available: {torch.cuda.is_available()}')
    logger.info(f'device: {device}')

    device_save = torch.device('cpu')
    model = torch.nn.DataParallel(model)
    model.to(device)

    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    out = []
    index = []
    valid = []


    for data_batch in tqdm(dl, desc='transforming images to vectors'):
        with torch.autograd.no_grad():
            x = data_batch[0]
            x = x.to(device)
            z = model(x).to(device_save)
            out.append(z)
            index.append(data_batch[1].to(device_save))
            valid.append(data_batch[2].to(device_save))
        
    vectors = torch.cat(out, dim=0).numpy()
    valid = torch.cat(valid, dim=0).numpy()

    df = pd.DataFrame({'path': images, 'valid': valid})
    print('valid / total', df['valid'].sum(), df.shape[0])
    df.to_csv(odir / 'image-valid.csv', index=False)
    np.save(odir / 'image-vectors.npy', vectors)
    

if __name__ == '__main__':
    main()



