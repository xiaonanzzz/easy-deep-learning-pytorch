import tqdm

from easydl import to_numpy
from easydl.config import TrainingConfig, RuntimeConfig
import torch
import numpy as np
import easydl


class LossAverage(object):
    def __init__(self):
        self.losses = []

    def append(self, loss):
        self.losses.append(loss.data.cpu().numpy())

    def mean(self):
        return float(np.mean(self.losses))


def forward_backward_one_step(model, criterion, x, y, opt: torch.optim.Optimizer, train_cfg:TrainingConfig, run_cfg: RuntimeConfig, store={}):
    model.train()
    model.to(run_cfg.device)
    criterion.to(run_cfg.device)

    assert torch.isnan(x).sum() == 0
    o = model(x.to(run_cfg.device))
    loss = criterion(o, y.to(run_cfg.device))
    opt.zero_grad()
    loss.backward()

    assert torch.isnan(loss).sum() == 0
    for pg in opt.param_groups:
        torch.nn.utils.clip_grad_value_(pg['params'], train_cfg.clip_gradient)

    # must do step
    opt.step()

    if 'output' in store:
        store['output'] = o
    return loss


def on_epoch_end(opt: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, run_cfg: RuntimeConfig,
                 metric_logger: easydl.MetricLogger, loss_avg=None, pbar=None):
    if run_cfg.print_verbose > 0:
        print('last learning rate', scheduler.get_last_lr())

    if pbar is not None:
        pbar.close()

    if loss_avg:
        metric_logger.log({metric_logger.TrainLossMovingAverage: loss_avg.mean()})
    for idx, lr in enumerate(scheduler.get_last_lr()):
        metric_logger.log({'{}_{}'.format(metric_logger.LastLr, idx): float(lr)})

    # perform must do actions
    scheduler.step()


class TrainAccuracyAverage():
    def __init__(self):
        self.prediction = []
        self.truth = []
        self.scores = []

    def update(self, pred, truth):
        """
        pred: torch.Tensor N*M, N*1, N,
        truth: torch.Tensor  N
        """
        truth = to_numpy(truth)
        assert truth.ndim == 1
        pred = np.squeeze(to_numpy(pred))
        assert pred.ndim == 2 or pred.ndim == 1
        self.truth.extend(truth)
        if pred.ndim == 2:
            pred = pred.argmax(axis=1)
        self.prediction.extend(pred)

    def accuracy(self):
        pred = np.array(self.prediction)
        truth = np.array(self.truth)
        return (pred == truth).mean()


def prepare_optimizer(args: TrainingConfig, param_groups):
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, momentum=args.momentum,
                              nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay=args.weight_decay,
                                  momentum=args.momentum)
    elif args.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)

    return opt


def prepare_lr_scheduler(args: TrainingConfig, opt):
    if args.lr_scheduler_type == 'step':
        lr = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler_type == 'cosine':
        lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.train_epoch, eta_min=args.lr_min)
    else:
        raise NotImplementedError()

    return lr