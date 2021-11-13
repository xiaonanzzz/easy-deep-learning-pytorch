from easydl.models.linear import LinearEmbedder
import torch
from easydl.trainer.metric_learning import ProxyAnchorLossEmbeddingModelTrainer, EpochEndEvaluationHook, ProxyAnchorLossConfigContainer


def debug_proxy_anchor_loss_algo():
    model = LinearEmbedder(300, 200)
    x = torch.randn(100, 300)
    y = torch.randint(0, 10, (100, ))
    print('data shape', x.shape, y.shape,)
    dataset = torch.utils.data.TensorDataset(x, y)

    cm = ProxyAnchorLossConfigContainer()
    cm.loss_config.epoch = 10
    print(cm.all_config_dict())
    trainer = ProxyAnchorLossEmbeddingModelTrainer(model, dataset, 200, 10, configure_manager=cm)
    trainer.epoch_end_hook = EpochEndEvaluationHook(model, dataset, configure_manager=cm)

    trainer.train()

def debug_retrieval_accuracy():

    from easydl.trainer.metrics import RetrivalMetrics
    qx = torch.FloatTensor([[0, 1,], [1, 1], [1, 0]])
    qy = torch.LongTensor([0, 1, 2])

    met = RetrivalMetrics(qx, qy, qx, qy, ignore_k=1)
    print('recall@1', met.recall_k(1))
    print('recall@2', met.recall_k(2))


if __name__ == '__main__':
    debug_proxy_anchor_loss_algo()