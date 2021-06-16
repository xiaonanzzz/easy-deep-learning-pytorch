import torch


class LRSchedulerArgs():
    def __init__(self):
        self.lr_decay_step = 10
        self.lr_decay_gamma = 0.5

def prepare_lr_scheduler(args, opt):
    lr = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    return lr