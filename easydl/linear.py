import torch
from torch import nn


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