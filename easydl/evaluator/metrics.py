from easydl.config import DeviceConfig
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

class RetrivalMetricsBase(object):

    def recall_k(self, k):
        pass

class RetrivalMetrics(DeviceConfig):
    def __init__(self, qx, qy, ix, iy, metric='cosine', ignore_k=0):
        super(RetrivalMetrics, self).__init__()
        assert qx.shape[1] == ix.shape[1]
        assert qx.shape[0] == qy.shape[0]
        assert ix.shape[0] == iy.shape[0]
        if metric == 'cosine':
            self.dist_mat = torch.matmul(F.normalize(qx, p=2),
                                         F.normalize(ix, p=2).T)
        else:
            raise NotImplemented
        self.qy = qy
        self.iy = iy
        self.ignore_k = ignore_k

    def recall_k(self, k):
        assert self.ignore_k + k <= self.dist_mat.shape[1]
        total = self.dist_mat.shape[0]
        topk_idx = torch.argsort(self.dist_mat, descending=True)
        topk_idx = topk_idx[:, self.ignore_k:self.ignore_k + k]
        topk_ys = self.iy[topk_idx]
        qy_ext = torch.unsqueeze(self.qy, dim=1)
        match_mat = qy_ext == topk_ys
        match_mat = match_mat.sum(dim=1) > 0
        match = match_mat.sum().item()

        return float(match) / float(total)

class RetrivalMetricsSklearn(DeviceConfig):
    key_match_mat = 'match_mat'
    key_dist = 'dist'

    def __init__(self, qx, qy, ix, iy, *args, ignore_k=0, algorithm='ball_tree', n_jobs=4, **kwargs):
        super(RetrivalMetricsSklearn, self).__init__(*args, **kwargs)
        assert qx.shape[1] == ix.shape[1]
        assert qx.shape[0] == qy.shape[0]
        assert ix.shape[0] == iy.shape[0]
        self.index = NearestNeighbors(algorithm=algorithm, n_jobs=n_jobs, **kwargs)
        self.index.fit(np.array(ix, dtype=np.float32))

        self.qx = np.array(qx, dtype=np.float32)
        self.qy = np.array(qy, dtype=np.int64)
        self.iy = np.array(iy, dtype=np.int64)
        self.ignore_k = ignore_k
        self.index_n = ix.shape[0]

    def recall_k(self, k, store_dict={}):
        assert self.ignore_k + k <= self.index_n, self.index_n
        total = self.qy.shape[0]

        topk_dist, topk_idx = self.index.kneighbors(self.qx, n_neighbors=self.ignore_k + k)
        topk_idx = topk_idx[:, self.ignore_k:self.ignore_k + k]
        topk_ys = self.iy[topk_idx]
        if self.key_dist in store_dict:
            store_dict[self.key_dist] = topk_dist[:, self.ignore_k:self.ignore_k + k]

        qy_ext = np.expand_dims(self.qy, axis=1)
        match_mat = qy_ext == topk_ys
        if self.key_match_mat in store_dict:
            store_dict[self.key_match_mat] = match_mat

        match_mat = match_mat.sum(axis=1) > 0
        match = match_mat.sum()
        acc = float(match) / float(total)

        return acc