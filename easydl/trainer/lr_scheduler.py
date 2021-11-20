import torch
from easydl.config import TrainingConfig

def prepare_lr_scheduler(args: TrainingConfig, opt):
    if args.lr_scheduler_type == 'step':
        lr = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler_type == 'cosine':
        lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.train_epoch)
    else:
        raise NotImplementedError()

    return lr


