import torch


def prepare_lr_scheduler(args, opt):
    lr = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    return lr


class LRSchedulerArgs(object):
    def __init__(self, *args, lr_decay_step=10, lr_decay_gamma=0.5, **kwargs):
        super(LRSchedulerArgs, self).__init__(*args, **kwargs)
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma