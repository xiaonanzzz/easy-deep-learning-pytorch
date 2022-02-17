from easydl.config import TrainingConfig, RuntimeConfig
import torch
import numpy as np


class LossAverage(object):
    def __init__(self):
        self.losses = []

    def append(self, loss):
        self.losses.append(loss.data.cpu().numpy())

    def mean(self):
        return float(np.mean(self.losses))


def forward_backward_one_step(model, criterion, x, y, opt, train_cfg:TrainingConfig, run_cfg: RuntimeConfig, store={}):
    model.train()
    model.to(run_cfg.device)
    criterion.to(run_cfg.device)

    assert torch.isnan(x).sum() == 0
    o = model(x.to(run_cfg.device))
    loss = criterion(o, y.to(run_cfg.device))
    opt.zero_grad()
    loss.backward()

    assert torch.isnan(loss).sum() == 0
    torch.nn.utils.clip_grad_value_(model.parameters(), train_cfg.clip_gradient)
    torch.nn.utils.clip_grad_value_(criterion.parameters(), train_cfg.clip_gradient)

    opt.step()

    if 'output' in store:
        store['output'] = o
    return loss
