import torch
from .optimizer import prepare_optimizer, OptimizerArgs
from .lr_scheduler import prepare_lr_scheduler, LRSchedulerArgs
from tqdm import tqdm
import numpy as np
from easydl.config import DeviceConfig


class ImageClassificationTrainer(DeviceConfig):
    def __init__(self):
        super(ImageClassificationTrainer, self).__init__()
        self.optimizer_args = OptimizerArgs()
        self.lr_scheduler_args = LRSchedulerArgs()
        self.batch_size = 32
        self.nb_epochs = 30
        self.nb_workers = 8 if torch.cuda.is_available() else 0

    def train(self, model, train_dataset, epoch_end_hook=None):
        args = self

        criterion = torch.nn.CrossEntropyLoss()

        param_groups = [{'params': model.parameters(), 'lr': float(args.optimizer_args.lr) * 1}]

        opt = prepare_optimizer(self.optimizer_args, param_groups)

        scheduler = prepare_lr_scheduler(self.lr_scheduler_args, opt)

        print("Training parameters: {}".format(vars(args)))

        dl_tr = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=args.nb_workers,
            pin_memory=True
        )

        for epoch in range(1, args.nb_epochs + 1):
            losses_per_epoch = []

            model.train()
            model.to(self.device)
            criterion.to(self.device)
            pbar = tqdm(enumerate(dl_tr))
            for batch_idx, (x, y) in pbar:
                assert torch.isnan(x).sum() == 0

                m = model(x.to(self.device))
                loss = criterion(m, y.to(self.device))

                opt.zero_grad()
                loss.backward()

                if torch.isnan(loss).sum() > 0:
                    print('model input\n', x)
                    print('embedding output\n', m)
                    print('label', y)
                    raise RuntimeError()

                torch.nn.utils.clip_grad_value_(model.parameters(), 10)

                losses_per_epoch.append(loss.data.cpu().numpy())
                opt.step()

                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(dl_tr),
                               100. * batch_idx / len(dl_tr),
                        np.mean(losses_per_epoch)))
            pbar.close()
            scheduler.step()

            if epoch_end_hook is not None:
                print('epoch done. calling hook function')
                epoch_end_hook(locals=locals())