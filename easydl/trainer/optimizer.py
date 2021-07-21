import torch

class OptimizerArgs(object):
    def __init__(self, *args, optimizer='sgd', lr=0.1, weight_decay=1e-4, momentum=0.9, **kwargs):
        super(OptimizerArgs, self).__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum



def prepare_optimizer(args:OptimizerArgs, param_groups):
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay=args.weight_decay,
                                  momentum=args.momentum)
    elif args.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)

    return opt