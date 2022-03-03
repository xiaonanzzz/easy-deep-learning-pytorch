import wandb

from easydl.experiments import MetricLogger
from easydl import RuntimeConfig, save_model
from easydl.models.mlp_model import LinearClassifier


def test_wandb():
    cfg = RuntimeConfig()

    metric_logger = MetricLogger(cfg)

    save_path = metric_logger.get_path('linear_clf.torch')

    model = LinearClassifier(32, 10)
    save_model(model, save_path)

    wandb.finish()


if __name__ == '__main__':
    test_wandb()