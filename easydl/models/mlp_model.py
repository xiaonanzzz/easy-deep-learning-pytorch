import torch
from torch import nn
from torch.nn.functional import normalize

"""
Attention: Try to depend only on torch, so that it's easy to export these models
"""

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class L2Normalization(torch.nn.Module):
    def __init__(self, dim=1):
        super(L2Normalization, self).__init__()
        self.dim = dim

    def forward(self, x):
        return normalize(x, p=2, dim=self.dim)


class L2NormEmbedder(torch.nn.Module):
    def __init__(self, model, dim=1):
        super(L2NormEmbedder, self).__init__()
        self.model = model
        self.dim = dim

    def forward(self, x):
        x = self.model(x)
        return normalize(x, p=2, dim=self.dim)


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        if self.training:
            return x
        else:
            return torch.argmax(x, dim=1)


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
        o = normalize(o)
        return o


class EmbedderClassifier(nn.Module):
    def __init__(self, embedder, embedding_size, num_classes):
        super(EmbedderClassifier, self).__init__()
        self.embedder = embedder
        self.classifier = LinearClassifier(embedding_size, num_classes)

    def forward(self, x):
        x = self.embedder(x)
        x = self.classifier(x)
        return x