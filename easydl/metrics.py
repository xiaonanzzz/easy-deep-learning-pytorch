from easydl.config import RuntimeConfig
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from easydl.batch_processing import batch_process_x_y_dataset_and_concat

class RetrivalMetricsBase(object):

    def recall_k(self, k):
        pass


def recall_in_k_pytorch(query_x, query_y, index_x, index_y, k_list=1, metric='cosine', ignore_k=0):
    qx, qy = query_x, query_y
    ix, iy = index_x, index_y
    assert qx.shape[1] == ix.shape[1] and qx.shape[0] == qy.shape[0] and ix.shape[0] == iy.shape[0]

    if metric == 'cosine':
        dist_mat = torch.matmul(F.normalize(qx, p=2),
                                     F.normalize(ix, p=2).T)
    else:
        raise NotImplemented

    def recall_k(k):
        assert ignore_k + k <= dist_mat.shape[1]
        total = dist_mat.shape[0]
        topk_idx = torch.argsort(dist_mat, descending=True)
        topk_idx = topk_idx[:, ignore_k:ignore_k + k]
        topk_idx = topk_idx
        topk_ys = iy[topk_idx]
        qy_ext = torch.unsqueeze(qy, dim=1)
        match_mat = qy_ext == topk_ys
        match_mat = match_mat.sum(dim=1) > 0
        match = match_mat.sum().item()
        return float(match) / float(total)
    if isinstance(k_list, int):
        return recall_k(k_list)
    elif type(k_list) in [list, tuple]:
        return {k: recall_k(k) for k in k_list}
    else:
        raise TypeError('Ks type not supported')


def recall_in_k_self_retrieval(model, testds, k_list):
    """
    return: a dictionary of { k (int): recall_at_k (float) }
    """

    # calculate embeddings with model and get targets
    model.eval()
    print('processing input datasets to get embedding and labels...')
    x, y = batch_process_x_y_dataset_and_concat(testds, model, tqdm_disable=False)

    # because it's self retrieval, so ignore the first one
    print('calulating recall...')
    ret = recall_in_k_pytorch(x, y, x, y, k_list=k_list, ignore_k=1)

    return ret


def recall_in_k_sklearn(query_x, query_y, index_x, index_y, ks=1, metric='cosine', ignore_k=0, n_jobs=4, store=None):
    qx, qy = query_x, np.array(query_y)
    ix, iy = index_x, np.array(index_y)

    index = NearestNeighbors(metric=metric, n_jobs=n_jobs)
    index.fit(np.array(ix, dtype=np.float32))

    def recall_k(k):
        assert ignore_k + k <= index.n_samples_fit_, index.n_samples_fit_
        total = qy.shape[0]

        topk_dist, topk_idx = index.kneighbors(np.array(qx), n_neighbors=ignore_k + k)
        topk_idx = topk_idx[:, ignore_k:ignore_k + k]
        topk_ys = np.take(iy, topk_idx)

        qy_ext = np.expand_dims(qy, axis=1)
        match_mat = (qy_ext == topk_ys)

        match_mat = np.sum(match_mat, axis=1) > 0
        match = np.sum(match_mat)
        acc = float(match) / float(total)
        return acc

    if isinstance(ks, int):
        return recall_k(ks)
    elif type(ks) in [list, tuple]:
        return {k: recall_k(k) for k in ks}
    else:
        raise TypeError('Ks type not supported')
