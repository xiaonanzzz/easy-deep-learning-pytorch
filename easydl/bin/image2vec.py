import argparse
import os
from easydl.models.convnext import convnext_base
from easydl.models.mlp_model import L2Normalization
import pandas as pd
from datetime import datetime
from easydl.image_transform import resnet_transform_test
from easydl.datasets.image_dataset import PureImageDataset
from easydl import batch_process_x_y_dataset_and_concat
import json

def read_image(args):
    if args.image.endswith('.csv'):
        df = pd.read_csv(args.image)
    

    if args.debug:
        df = df.head(10)

    return df
    

def main():
    parser = argparse.ArgumentParser(description="Transform images into vectors")
    parser.add_argument("--model", help="confidence threshold", default='convnext-base', type=str)
    parser.add_argument("--image", help="Image to be predicted", type=str)
    parser.add_argument('--debug', type=int, default=0, help='debug mode, only run first few samples')
    parser.add_argument('--abs_path', type=str, default='abs_path', help='the column name of absolute image path')
    parser.add_argument('--append_data_to_csv', '-a', type=int, default=1, help='if 1, it will save the vectors as str and append it to the csv file')
    parser.add_argument('--round', type=int, default=6, help='the number of decimal places you want to round to, -1 represents no round')
    parser.add_argument('-o', '--out_dir', type=str, default='image2vec-{}'.format(str(datetime.today()).replace(' ', '-')))
    parser.add_argument('--batch_size', type=int, default=200, help='the batch size for splitting the original data')
    args = parser.parse_args()

    abs_path = args.abs_path

    df = read_image(args)
    assert abs_path in df.columns, 'cannot find the column of image file'
    print(f'### load images ... {len(df)} ')

    if args.model == 'convnext-base':
        model = convnext_base(pretrained=True, in_22k=True, num_classes=21841)
        model.head = L2Normalization()
        dataset = PureImageDataset(df[abs_path], resnet_transform_test)
    else: 
        raise NotImplemented


    vectors, indexes = batch_process_x_y_dataset_and_concat(dataset, model, tqdm_disable=False, to_numpy=True)

    assert 2 == len(vectors.shape), 'the vectors should be a 2d array'


    #### output results ####
    os.makedirs(args.out_dir, exist_ok=True)

    if args.append_data_to_csv:
        df['image2vec'] = None
        for vec, idx in zip(vectors, indexes):
            if args.round > -1:
                vec = [round(x, args.round) for x in vec.tolist()]
            else:
                vec = vec.tolist()
            df['image2vec'][idx] = json.dumps(vec)
        
        df.to_csv(os.path.join(args.out_dir, 'image2vec.csv'), index=False)
    else:
        raise NotImplemented


if __name__ == '__main__':
    main()



