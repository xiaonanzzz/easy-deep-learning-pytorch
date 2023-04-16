
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
from glob import glob


def main():

    """
    Input: dir
        e.g. dir/0/ dir/1/ dir/2/
        they will be merged into one folder
        out/image-valid.csv     # the image 2 vector meta data csv
        out/image-vectors.npy   # the vectors numpy array

    """
    parser = argparse.ArgumentParser(description="Merge vectors into one file")
    parser.add_argument('-f', "--rootf", type=str, required=True, help="dir which contains vectors from image2vec")
    parser.add_argument('-o', '--out_dir', type=str, default='tmp/vec-merge-{}'.format(str(datetime.today()).replace(' ', '-')))
    parser.add_argument('-e', '--exclude_invalid', type=int, default=1, )
    
    parser.print_help()
    args = parser.parse_args()

    print('Input args: ', args.__dict__)
    metas, vecs = [], []
    root_dirs = list(glob(f"{args.rootf}/*/", recursive = True))
    for d in tqdm(root_dirs, 'processing dirs'):
        metafp = Path(d) / 'image-valid.csv'
        vecfp = Path(d) / 'image-vectors.npy'

        if metafp.exists() and vecfp.exists():
            dfmx = pd.read_csv(metafp)
            vec = np.load(vecfp)

            assert dfmx.shape[0] == vec.shape[0], (dfmx.shape, vec.shape)
            metas += [dfmx]
            vecs += [vec]

    dfma =pd.concat(metas, axis=0)
    veca = np.concatenate(vecs, axis=0)
    assert dfma.shape[0] == veca.shape[0]

    if args.exclude_invalid:
        valid = dfma['valid']
        dfma = dfma[valid]
        veca = veca[valid]

        assert dfma.shape[0] == veca.shape[0]

    print('merged vector shape: ', veca.shape)

    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    dfma.to_csv(odir / 'image-valid.csv', index=False)
    np.save(odir / 'image-vectors.npy', veca)




if __name__ == '__main__':
    main()