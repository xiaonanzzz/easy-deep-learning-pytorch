import torch

class OptimizerArgs(object):
    def __init__(self):
        super(OptimizerArgs, self).__init__()
        self.optimizer = 'sgd'
        self.lr = 0.1
        self.weight_decay = 1e-4
        self.momentum = 0.9



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