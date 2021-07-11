from torch import nn
import torch
import torch.nn.functional as F

class L2Normalization(nn.Module):
    def __init__(self, dim=1):
        super(L2Normalization, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
