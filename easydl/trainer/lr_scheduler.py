import torch
from easydl.config import LRSchedulerConfig

def prepare_lr_scheduler(args: LRSchedulerConfig, opt):
    lr = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    return lr


