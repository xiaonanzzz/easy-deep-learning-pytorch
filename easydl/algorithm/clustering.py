import os
import pandas as pd
from sklearn.cluster import MeanShift, AgglomerativeClustering, KMeans
from sklearn.metrics import homogeneity_completeness_v_measure, fowlkes_mallows_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter

from easydl.common import prepare_path

KEY_CLUSTERS = 'clusters'
KEY_CLASSES = 'classes'
KEY_AMI_SCORE = 'adjusted_mutual_info_score'
KEY_FM_SCORE = 'fowlkes_mallows_score'
KEY_PRECISION = 'precision'
KEY_PURERATE = 'pure_rate'
KEY_CLASS_TO_CLUSTER_RATIO = 'class_to_cluster_raio'
KEY_PURE_RATE_TIMES_CCRATIO = 'pure_times_class_cluster_ratio'

def evaluate_clustering_with_labels(ytrue, ycluster):
    true_label_encoder = LabelEncoder()
    cluster_label_encoder = LabelEncoder()

    ytrue_int = true_label_encoder.fit_transform(ytrue)
    ycluster_int = cluster_label_encoder.fit_transform(ycluster)

    result = {}
    result['clusters'] = len(set(ycluster_int))
    result['classes'] = len(set(ytrue_int))
    result['adjusted_mutual_info_score'] = adjusted_mutual_info_score(ytrue_int, ycluster_int)
    result['fowlkes_mallows_score'] = fowlkes_mallows_score(ytrue_int, ycluster_int)

    num_correct = 0
    pure_count = 0
    prediction = {}
    for k, v in zip(ycluster_int, ytrue_int):
        if k not in prediction:
            prediction[k] = Counter()
        prediction[k].update([v])
    for k, c in prediction.items():
        num_correct += c.most_common(1)[0][1]
        if sum(c.values()) == c.most_common(1)[0][1]:
            pure_count += sum(c.values())
    result['precision'] = num_correct / len(ytrue_int)
    result['pure_rate'] = pure_count / len(ytrue_int)
    result['class_to_cluster_raio'] = len(set(ytrue_int)) / len(set(ycluster_int))
    result['pure_times_class_cluster_ratio'] = result['pure_rate'] * result['class_to_cluster_raio']

    return result


def tune_mean_shift(x, y,  bandwidth_range=None, save_path='meanshift-tune.csv', disable_tqdm=False):
    """
    x, y are testing set
    """
    from tqdm import tqdm

    rows = []
    if bandwidth_range is None:
        bandwidth_range = np.arange(0.05, 1, 0.05)

    for bandwidth in tqdm(bandwidth_range, disable=disable_tqdm):
        cls = MeanShift(bandwidth=bandwidth)
        ypred = cls.fit_predict(x)
        metrics1 = evaluate_clustering_with_labels(y, ypred)

        row = {'bandwidth': bandwidth}
        row.update(metrics1)

        rows.append(row)

    rows = pd.DataFrame(rows)
    prepare_path(save_path)
    rows.to_csv(save_path, index=False)


def tune_ward_clustering(x, y, save_dir='', threshold_list=None, disable_tqdm=False):
    # x, y are numpy arrays
    from tqdm import tqdm

    thresholds = threshold_list or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2, 2.5, 3, 4]

    rows = []
    for th in tqdm(thresholds, disable=disable_tqdm):
        cls = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=th)
        ypred = cls.fit_predict(x)
        metrics1 = evaluate_clustering_with_labels(y, ypred, )

        row = {'threshold': th}
        row.update(metrics1)

        rows.append(row)

    rows = pd.DataFrame(rows)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        rows.to_csv(os.path.join(save_dir, 'ward-clustering-tunning.csv'), index=False)
    else:
        print(rows)


def tune_kmeans(x, y, save_dir='', k_list=None, disable_tqdm=False):
    # x, y are numpy arrays
    from tqdm import tqdm

    ks = k_list or [50, 77, 100, 150, 177, 200, 300]

    rows = []
    for k in tqdm(ks, disable=disable_tqdm, desc='tune kmeans'):
        cls = KMeans(n_clusters=k)
        ypred = cls.fit_predict(x)
        metrics1 = evaluate_clustering_with_labels(y, ypred, )

        row = {'k': k}
        row.update(metrics1)

        rows.append(row)

    rows = pd.DataFrame(rows)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        rows.to_csv(os.path.join(save_dir, 'kmeans-clustering-tunning.csv'), index=False)
    else:
        print(rows)