import torch
from torch import nn
from easydl.models import L2Normalization



class MLPEmbedder(nn.Module):
    def __init__(self, feature_dims, input_normalize=None, output_normalize=None, bias=False):
        super(MLPEmbedder, self).__init__()

        layers = []
        if input_normalize is not None:
            layers += [self.get_normalize_layer(input_normalize, feature_dims[0])]
        for i in range(len(feature_dims)-1):
            layers += [nn.Linear(feature_dims[i], feature_dims[i+1], bias=bias)]
            if i < len(feature_dims) - 2:
                layers += [nn.ReLU()]
        if output_normalize is not None:
            layers += [self.get_normalize_layer(output_normalize, feature_dims[-1])]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_normalize_layer(self, normalize_name, norm_size=None):
        if normalize_name == 'l2':
            return L2Normalization(dim=1)
        elif normalize_name == 'layer_norm':
            return nn.LayerNorm(norm_size)
        elif normalize_name == 'batch_norm':
            return nn.BatchNorm1d(norm_size)


