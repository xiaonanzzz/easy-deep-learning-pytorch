import torch
from easydl.datasets.cub import Cub2011MetricLearningDS
from easydl.trainer.metric_learning import ProxyAnchorLossEmbeddingModelTrainer, EpochEndEvaluationHook
from torch import nn
from easydl.models import L2Normalization

class SimpleNet(nn.Module):

    def __init__(self, out_dim=32) -> None:
        super(SimpleNet, self).__init__()
        self.channels = 64
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.channels, out_dim)
        self.out_norm = L2Normalization(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)     # batch, channels, 1, 1 -> batch, channels
        x = self.fc(x)
        return x

def simple_net_example():
    import os
    print('working directory', os.getcwd())

    train_data = Cub2011MetricLearningDS(os.path.expanduser('~/data/CUB_200_2011'), split='train')
    test_data = Cub2011MetricLearningDS(os.path.expanduser('~/data/CUB_200_2011'), split='test')

    print('data shape', train_data[0][0].shape, train_data[0])
    embsize = 32
    model = SimpleNet(embsize)

    trainer = ProxyAnchorLossEmbeddingModelTrainer(model, train_data, embsize, 100, tqdm_disable=False)
    trainer.epoch_end_hook = EpochEndEvaluationHook(model, test_data)
    trainer.train()

def resnet50_example():
    import os
    from easydl.models.cnn_embedder import Resnet50PALVersion
    print('working directory', os.getcwd())

    train_data = Cub2011MetricLearningDS(os.path.expanduser('~/data/CUB_200_2011'), split='train')
    test_data = Cub2011MetricLearningDS(os.path.expanduser('~/data/CUB_200_2011'), split='test')

    print('data shape', train_data[0][0].shape, train_data[0])
    embsize = 32
    model = Resnet50PALVersion(embsize)

    trainer = ProxyAnchorLossEmbeddingModelTrainer(model, train_data, embsize, 100, tqdm_disable=False)
    trainer.epoch_end_hook = EpochEndEvaluationHook(model, test_data)
    trainer.train()

if __name__ == '__main__':
    resnet50_example()

