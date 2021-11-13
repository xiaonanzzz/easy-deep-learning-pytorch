import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.metrics import homogeneity_completeness_v_measure, fowlkes_mallows_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter

from easydl import prepare_path

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