import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import easydl
from modulefinder import Module
from easydl.algorithm.image_classification import train_image_classification_model_2021_nov
from easydl.algorithm.metric_learning import *
from easydl.algorithm.clustering import tune_ward_clustering
from easydl.config import RuntimeConfig, TrainingConfig, get_config_from_cmd
from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.experiments import MetricLogger
from easydl.image_transform import timm_image_transform_imagenet_default
from easydl.models.image_model import Resnet50PALVersion
from easydl.models.mlp_model import EmbedderClassifier, LinearEmbedder, L2Normalization
from easydl.models.convnext import convnext_base, get_convnext_version_augmentation_config
from easydl.batch_processing import batch_process_x_y_dataset_and_concat

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_auc_score

if __name__ == '__main__':

    def compute_pos_neg_distribution(model_file):
        device = 'cuda:0'
        model = convnext_base(pretrained=True, in_22k=True, num_classes=21841) 
        # replace the head with a linear embedder
        model.head = L2Normalization()

        model.load_state_dict(torch.load(ROOT / model_file))
        model.to(device)
        model.eval()
        
        cub_exp = CubMetricLearningExperiment()
        test_ds = cub_exp.get_test_ds(cub_exp.testing_transform)    # using default transform

        x, y = batch_process_x_y_dataset_and_concat(test_ds, model, disable_tqdm=False)

        print(x.shape, y.shape)

        # get cosine distance
        x = x.to(device)
        x_norm = F.normalize(x, p=2)
        with torch.no_grad():
            cosine_dist = torch.matmul(x_norm, x_norm.T)
            cosine_dist = easydl.to_numpy(cosine_dist)
        print('distance done', cosine_dist.shape)
        distances_list = []
        n = x.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                is_same = (y[i] == y[j])
                distances_list.append((cosine_dist[i][j], is_same))
        
        distances_pos = [x[0] for x in distances_list if x[1]]
        distances_neg = [x[0] for x in distances_list if not x[1]]
        print('pos and neg distances done...')

        print('pos mean and std', np.mean(distances_pos), np.std(distances_pos))
        print('negative mean and std', np.mean(distances_neg), np.std(distances_neg))

    
    compute_pos_neg_distribution('dml_exp/runs/convnext_cub/lilac-deluge-22/best_model.torch')
    compute_pos_neg_distribution('dml_exp/runs/convnext_cub/lilac-deluge-22/last_model.torch')



    











