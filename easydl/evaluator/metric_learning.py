from easydl.config import RuntimeConfig, ConfigConsumer
from easydl.utils import batch_process_x_y_dataset_and_concat
import torch
from .metrics import RetrivalMetrics, RetrivalMetricsSklearn


class MetricLearningModelEvaluatorSingleSet(ConfigConsumer):
    def __init__(self, test_dataset, k_values=None, metric_pyclass=RetrivalMetricsSklearn, **kwargs):
        super(MetricLearningModelEvaluatorSingleSet, self).__init__(**kwargs)
        self.test_dataset = test_dataset
        self.metric_pyclass = metric_pyclass
        k_values = k_values or [1]
        self.recall_at_k = {k: 0 for k in k_values}

        self.check_config([RuntimeConfig])

    def evaluate(self, model):
        runcfg: RuntimeConfig = self.configure_container[RuntimeConfig]
        x, y = batch_process_x_y_dataset_and_concat(self.test_dataset, model,
                                                    batch_size=runcfg.infer_batch_size,
                                                    save_index=1,
                                                    device=runcfg.device)

        metrics = self.metric_pyclass(x, y, x, y, ignore_k=1)

        for k in self.recall_at_k:
            self.recall_at_k[k] = metrics.recall_k(k)

