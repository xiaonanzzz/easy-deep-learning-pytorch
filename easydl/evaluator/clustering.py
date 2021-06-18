from sklearn.metrics import homogeneity_completeness_v_measure, fowlkes_mallows_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter


def evaluate_clustering_with_labels(ytrue, ycluster):
    true_label_encoder = LabelEncoder()
    cluster_label_encoder = LabelEncoder()

    ytrue_int = true_label_encoder.fit_transform(ytrue)
    ycluster_int = cluster_label_encoder.fit_transform(ycluster)

    result = {}
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

    return result

