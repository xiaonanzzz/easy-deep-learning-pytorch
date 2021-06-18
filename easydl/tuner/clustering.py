from sklearn.cluster import MeanShift
import numpy as np
from easydl.evaluator.clustering import evaluate_clustering_with_labels


def kpi_purity_class_cluster_score(y, y_cluster):
    metrics = evaluate_clustering_with_labels(y, y_cluster)

    kpi = metrics['pure_rate'] * metrics['class_to_cluster_raio']

    return kpi


class ClusteringTuner():
    def __init__(self):
        self.kpi_function = kpi_purity_class_cluster_score

    def tune(self, x, y):
        pass


class MeanShiftTuner(ClusteringTuner):
    def __init__(self):
        super(MeanShiftTuner, self).__init__()
        self.bandwidth_set = np.arange(0, 1, 0.05)

    def tune(self, x, y):
        best_kpi = -np.inf
        for bandwidth in self.bandwidth_set:
            cluster = MeanShift(bandwidth=bandwidth)
            yc = cluster.fit_predict(x, y)

            kpi = self.kpi_function(y, yc)

            if kpi > best_kpi:
                best_kpi = kpi
                best_setting = {'algorithm': 'meanshift',
                                'bandwidth': bandwidth}
        return best_kpi, best_setting