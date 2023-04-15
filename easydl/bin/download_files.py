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
import urllib.request
from p_tqdm import p_map


def download_pid(row):
    try:
        url = row[0]
        image_path = row[1]
        urllib.request.urlretrieve(url, image_path)
        return 1
    except:
        #total_failed += 1
        return 0

def main():
    parser = argparse.ArgumentParser(description="Transform images into vectors")
    parser.add_argument("--input_csv", '-c', help="Image to be predicted", type=str)
    parser.add_argument('--save_root', default='tmp/images', type=str)
    parser.add_argument('--url_key', default='url')
    parser.add_argument('--path_key', default='path')
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('-o', '--save_csv', default='tmp/download-files-result-{}.csv'.format(str(datetime.today()).replace(' ', '-')))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    
    os.makedirs(args.save_root, exist_ok=True)
    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    items = [
        (row[0], os.path.join(args.save_root, row[1])) for row in df[[args.url_key, args.path_key]].itertuples(index=False)
    ]
    results = p_map(download_pid, items)
    df['success'] = results
    
    df.to_csv(save_path, index=False)
    print('success / total: ', df['success'].sum(), df.shape[0])
    

if __name__ == '__main__':
    main()



