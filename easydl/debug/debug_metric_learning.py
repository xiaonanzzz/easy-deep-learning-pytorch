from easydl.models.linear import LinearEmbedder
import torch
from easydl.trainer.metric_learning import ProxyAnchorLossEmbeddingModelTrainer


def debug_proxy_anchor_loss_algo():
    model = LinearEmbedder(300, 200)
    x = torch.randn(100, 300)
    y = torch.randint(0, 10, (100, ))
    print('data shape', x.shape, y.shape,)
    dataset = torch.utils.data.TensorDataset(x, y)

    trainer = ProxyAnchorLossEmbeddingModelTrainer()
    trainer.nb_epochs = 10
    trainer.train(model, dataset, 200, 10)

if __name__ == '__main__':
    debug_proxy_anchor_loss_algo()