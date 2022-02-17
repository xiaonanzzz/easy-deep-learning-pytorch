import numpy as np
from torch.utils.data.sampler import BatchSampler

from easydl.losses import ProxyAnchorLoss
from easydl.optimizer import prepare_optimizer
from easydl.lr_scheduler import prepare_lr_scheduler
from easydl.config import *
from easydl.training import forward_backward_one_step, LossAverage
from easydl.experiments import PrintMetricLogger
from tqdm import *


class ProxyAnchorLossConfig(ConfigBase):
    def __init__(self):
        self.embedding_size = 512
        self.margin = 0.1
        self.alpha = 32
        self.proxy_lr_scale = 100


def train_embedding_model_with_proxy_anchor_loss_v2(model, train_ds, nb_classes, metric_logger,
                                                    train_cfg: TrainingConfig, run_cfg: RuntimeConfig, loss_cfg: ProxyAnchorLossConfig,
                                                    epoch_end_hook=None):

    criterion = ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=loss_cfg.embedding_size, mrg=loss_cfg.margin, alpha=loss_cfg.alpha)

    param_groups = [{'params': model.parameters(), 'lr': float(train_cfg.lr) * 1},
                    {'params': criterion.proxies, 'lr': float(train_cfg.lr) * loss_cfg.proxy_lr_scale}]
    opt = prepare_optimizer(train_cfg, param_groups)
    scheduler = prepare_lr_scheduler(train_cfg, opt)

    dl_tr = torch.utils.data.DataLoader(train_ds, batch_size=train_cfg.train_batch_size, num_workers=run_cfg.cpu_workers, pin_memory=True)

    for epoch in range(1, train_cfg.train_epoch + 1):
        loss_avg = LossAverage()
        pbar = tqdm(enumerate(dl_tr), total=len(dl_tr), disable=run_cfg.tqdm_disable)
        for batch_idx, (x, y) in pbar:

            loss = forward_backward_one_step(model, criterion, x, y, opt, train_cfg, run_cfg)

            loss_avg.append(loss)

            pbar.set_postfix({'epoch': epoch, 'batch': batch_idx, 'total_batch': len(dl_tr),
                              'loss_mean': np.mean(loss_avg.mean())},
                             refresh=True)

        metric_logger.log({'training_loss': loss_avg.mean()})
        pbar.close()
        scheduler.step()

        if epoch_end_hook is not None:
            epoch_end_hook(locals=locals())