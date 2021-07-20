from easydl.config import DeviceConfig
import torch
import torch.nn.functional as F

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

        return match / total


