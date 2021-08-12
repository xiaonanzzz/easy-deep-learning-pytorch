
def check_RetrivalMetricsFaiss():
    import torch
    from easydl.evaluator.metrics import RetrivalMetricsSklearn
    qx = torch.FloatTensor([[0, 1, ], [1, 1], [1, 0]])
    qy = torch.LongTensor([0, 1, 2])

    met = RetrivalMetricsSklearn(qx, qy, qx, qy, ignore_k=1)
    assert met.recall_k(1) == 0.0
    assert met.recall_k(2) == 0.0

    met = RetrivalMetricsSklearn(qx, qy, qx, qy, ignore_k=0)
    assert met.recall_k(1) == 1
    assert met.recall_k(2) == 1

    print('check_RetrivalMetricsFaiss done')

def check_timer():
    from easydl.utils import TimerContext
    import time
    with TimerContext(name='timer 1'):
        time.sleep(0.1)

def check_batch_process():
    from easydl.utils import batch_process_tensor
    from easydl.models.mlp import MLPEmbedder
    import numpy as np
    model = MLPEmbedder([128, 64], output_normalize='l2')
    x = np.zeros((3, 128))
    y = batch_process_tensor(x, model)
    print(y.shape)
    assert y.shape == (3, 64)
    print('check passed', 'from easydl.utils import batch_process_tensor')

def fast_checks():
    check_timer()
    check_RetrivalMetricsFaiss()
    check_batch_process()

if __name__ == '__main__':
    fast_checks()