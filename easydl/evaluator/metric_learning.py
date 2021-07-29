from easydl.config import DeviceConfig
from easydl.utils import batch_process_x_y_dataset_and_concat
import torch
from .metrics import RetrivalMetrics, RetrivalMetricsSklearn


class MetricLearningModelEvaluatorSingleSet(DeviceConfig):
    def __init__(self, test_dataset, k_values=None, batch_size=32, metric_pyclass=RetrivalMetricsSklearn):
        super(MetricLearningModelEvaluatorSingleSet, self).__init__()
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.metric_pyclass = metric_pyclass
        k_values = k_values or [1]
        self.recall_at_k = {k: 0 for k in k_values}

    def evaluate(self, model):
        x, y = batch_process_x_y_dataset_and_concat(self.test_dataset, model, batch_size=self.batch_size, save_index=1, device=self.device)

        metrics = self.metric_pyclass(x, y, x, y, ignore_k=1)

        for k in self.recall_at_k:
            self.recall_at_k[k] = metrics.recall_k(k)
