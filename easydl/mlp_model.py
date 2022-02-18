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


class LinearEmbedder(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearEmbedder, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        o = self.fc(x)
        o = self.l2_norm(o)
        return o

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output