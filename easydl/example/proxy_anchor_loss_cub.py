import torch
from easydl.datasets.cub import Cub2011MetricLearningDS
from easydl.trainer.metric_learning import ProxyAnchorLossEmbeddingModelTrainer, EpochEndEvaluationHook
from torch import nn
from easydl.models import L2Normalization
from easydl.utils.experiments import WandbLogger, MetricLogger

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
    """
    reproducing the result from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 120 \
                --lr 1e-4 \
                --dataset cub \
                --warm 5 \
                --bn-freeze 1 \
                --lr-decay-step 5

    :return:
    """
    import os
    from easydl.models.cnn_embedder import Resnet50PALVersion
    print('working directory', os.getcwd())

    train_data = Cub2011MetricLearningDS(os.path.expanduser('~/data/CUB_200_2011'), split='train')
    test_data = Cub2011MetricLearningDS(os.path.expanduser('~/data/CUB_200_2011'), split='test')

    print('data shape', train_data[0][0].shape, train_data[0])
    embsize = 512
    model = Resnet50PALVersion(embsize)

    trainer = ProxyAnchorLossEmbeddingModelTrainer(model, train_data, embsize, 100, tqdm_disable=False)
    trainer.epoch_end_hook = EpochEndEvaluationHook(model, test_data)
    trainer.train()

def resnet50_example_no_pretrain_proxy_anchor_loss_paper(wandb_api=None):
    """
    reproducing the result from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
    python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 120 \
                --lr 1e-4 \
                --dataset cub \
                --warm 5 \
                --bn-freeze 1 \
                --lr-decay-step 5

    :return:
    """
    import os
    from easydl.models.cnn_embedder import Resnet50PALVersion
    print('working directory', os.getcwd())

    train_data = Cub2011MetricLearningDS(os.path.expanduser('~/data/CUB_200_2011'), split='train')
    test_data = Cub2011MetricLearningDS(os.path.expanduser('~/data/CUB_200_2011'), split='test')

    print('data shape', train_data[0][0].shape, train_data[0])
    embsize = 512
    model = Resnet50PALVersion(embsize, pretrained=False, bn_freeze=False)
    metric_logger = WandbLogger(project='cub-reproduce', tags=['resnet50', 'proxy-anchor-loss', ], api_key=wandb_api, prepare=True) if wandb_api else MetricLogger()
    trainer = ProxyAnchorLossEmbeddingModelTrainer(model, train_data, embsize, 100, tqdm_disable=False, metric_logger=metric_logger)
    trainer.epoch_end_hook = EpochEndEvaluationHook(model, test_data, metric_logger=metric_logger)

    trainer.train()

if __name__ == '__main__':
    resnet50_example_no_pretrain_proxy_anchor_loss_paper()

