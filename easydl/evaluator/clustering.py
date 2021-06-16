from sklearn.metrics import homogeneity_completeness_v_measure, fowlkes_mallows_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter

def evaluate_clustering_with_labels(ytrue, ycluster, metrics=['precision', 'class_to_cluster_raio', 'adjusted_mutual_info_score', 'fowlkes_mallows_score']):
    true_label_encoder = LabelEncoder()
    cluster_label_encoder = LabelEncoder()

    ytrue_int = true_label_encoder.fit_transform(ytrue)
    ycluster_int = cluster_label_encoder.fit_transform(ycluster)

    result = {

    }
    if 'adjusted_mutual_info_score' in metrics:
        result['adjusted_mutual_info_score'] = adjusted_mutual_info_score(ytrue_int, ycluster_int)
    if 'fowlkes_mallows_score' in metrics:
        result['fowlkes_mallows_score'] = fowlkes_mallows_score(ytrue_int, ycluster_int)
    if 'precision' in metrics:
        num_correct = 0
        prediction = {}
        for k, v in zip(ycluster_int, ytrue_int):
            if k not in prediction:
                prediction[k] = Counter()
            prediction[k].update([v])
        for k, c in prediction.items():
            num_correct += c.most_common(1)[0][1]
        result['precision'] = num_correct / len(ytrue_int)
    if 'class_to_cluster_raio' in metrics:
        result['class_to_cluster_raio'] = len(set(ytrue_int)) / len(set(ycluster_int))

    return result


