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
    
class ImageUrlDataset:
    def __init__(self, urls, transform=None):
        self.urls = list(urls)
        self.transform = transform

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, index):
        url = self.urls[index]

        try:
            im = Image.open(requests.get(url, stream=True).raw)
            valid = 1
        except:
            # create a black image
            im = Image.new(mode='RGB', size=(300, 300), color=0)
            valid = 0
        if self.transform is not None:
            im = self.transform(im)

        return (im, index, valid)
    

def _debug():

    urls = ['https://m.media-amazon.com/images/I/7105+qtR0ML.jpg', 'https://m.media-amazon.com/images/I/7105+qtR0MLsfs.jpg']
    ds = ImageUrlDataset(urls, transform=resnet_transform_test)
    print(ds[0], ds[1])

def main():
    parser = argparse.ArgumentParser(description="Transform images into vectors")
    parser.add_argument("--input_csv", '-c', help="Image to be predicted", type=str)
    parser.add_argument("--model", help="confidence threshold", default='convnext-base', type=str)
    parser.add_argument('--batch_size', type=int, default=200, help='the batch size for splitting the original data')
    parser.add_argument('--debug', type=int, default=0, help='debug mode, only run first few samples')
    parser.add_argument('-o', '--out_dir', type=str, default='url2vec-{}'.format(str(datetime.today()).replace(' ', '-')))
    args = parser.parse_args()

    if args.debug:
        _debug()

    # prepare run 
    batch_size = args.batch_size
    df = pd.read_csv(args.input_csv)
    urls = df['url'].tolist()
    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filemode=odir / 'url2vec.log', stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()

    if args.model == 'convnext-base':
        model = convnext_base(pretrained=True, in_22k=True, num_classes=21841)
        model.head = L2Normalization()
        dataset = ImageUrlDataset(urls, resnet_transform_test)
    else: 
        raise NotImplemented
    
    logger.info(f'total urls: {len(urls)}')


    # run model on all dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda')
    logging.info(f'cuda available: {torch.cuda.is_available()}')
    logger.info(f'device: {device}, {visible_devices}')
    device_save = torch.device('cpu')
    model = torch.nn.DataParallel(model)
    model.to(device)

    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True,
        prefetch_factor=4,
    )
    out = []
    index = []
    valid = []


    for data_batch in tqdm(dl, desc='transforming urls to vectors'):
        with torch.autograd.no_grad():
            x = data_batch[0]
            x = x.to(device)
            z = model(x).to(device_save)
            out.append(z)
            index.append(data_batch[1].to(device_save))
            valid.append(data_batch[2].to(device_save))
        
    vectors = torch.cat(out).numpy()
    valid = torch.cat()

    print(vectors.shape, valid)

if __name__ == '__main__':
    main()



