import numpy as np
from torch.utils.data.sampler import BatchSampler

import easydl
from easydl.losses import ProxyAnchorLoss
from easydl.config import *
from easydl.training_common import forward_backward_one_step, LossAverage, on_epoch_end, prepare_optimizer, \
    prepare_lr_scheduler
from easydl.mlp_model import LinearClassifier
from tqdm import *



class ProxyAnchorLossConfig(ConfigBase):
    def __init__(self, **kwargs):
        self.embedding_size = 512
        self.margin = 0.1
        self.alpha = 32
        self.proxy_lr_scale = 100

        self.from_dict(kwargs)


def train_embedding_model_with_proxy_anchor_loss_with_warmup_freeze(model, train_ds, nb_classes, metric_logger,
                                                    train_cfg: TrainingConfig, run_cfg: RuntimeConfig,
                                                    loss_cfg: ProxyAnchorLossConfig,
                                                    epoch_end_hook=None, freezing_params_during_warmup=None):
    """
    freezing_params_during_warmup:  None: no freezing, "model": freezing all model parameters, list(params): freezing the given list of params
    """
    print('configurations', train_cfg, loss_cfg)
    criterion = ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=loss_cfg.embedding_size, mrg=loss_cfg.margin, alpha=loss_cfg.alpha)

    if freezing_params_during_warmup is None:
        print('not freezing anything...')
        freezing_params_during_warmup = list()
    elif freezing_params_during_warmup == 'model':
        freezing_params_during_warmup = list(model.parameters())
        print('freezing all model parameters for warmup ...', len(freezing_params_during_warmup))
    else:
        print('freezing given model parameters, total: ', len(freezing_params_during_warmup))

    param_groups = [{'params': model.parameters(), 'lr': float(train_cfg.lr) * 1},
                    {'params': criterion.proxies, 'lr': float(train_cfg.lr) * loss_cfg.proxy_lr_scale}]
    opt = prepare_optimizer(train_cfg, param_groups)
    scheduler = prepare_lr_scheduler(train_cfg, opt)

    dl_tr = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=train_cfg.train_batch_size, num_workers=run_cfg.cpu_workers, pin_memory=True)

    for epoch in range(1, train_cfg.train_epoch + 1):
        loss_avg = LossAverage()

        for param in freezing_params_during_warmup:
            param.requires_grad = (epoch > train_cfg.warmup_epoch)

        pbar = tqdm(enumerate(dl_tr), total=len(dl_tr), disable=run_cfg.tqdm_disable)
        for batch_idx, (x, y) in pbar:

            loss = forward_backward_one_step(model, criterion, x, y, opt, train_cfg, run_cfg)

            loss_avg.append(loss)

            pbar.set_postfix({'epoch': epoch, 'batch': batch_idx, 'total_batch': len(dl_tr),
                              'loss_mean': np.mean(loss_avg.mean())},
                             refresh=True)

        on_epoch_end(opt, scheduler, run_cfg, metric_logger, loss_avg=loss_avg, pbar=pbar)

        if epoch_end_hook is not None:
            epoch_end_hook(locals=locals())


def train_embedder_proxy_anchor_loss_and_classification_loss(model, train_ds, nb_classes, metric_logger,
                                                    train_cfg: TrainingConfig, run_cfg: RuntimeConfig,
                                                    loss_cfg: ProxyAnchorLossConfig,
                                                    epoch_end_hook=None, freezing_params_during_warmup=None):
    """
    freezing_params_during_warmup:  None: no freezing, "model": freezing all model parameters, list(params): freezing the given list of params
    """
    print('configurations', train_cfg, loss_cfg)
    criterion = ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=loss_cfg.embedding_size, mrg=loss_cfg.margin, alpha=loss_cfg.alpha)
    clf_loss = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(loss_cfg.embedding_size, nb_classes)

    if freezing_params_during_warmup is None:
        print('not freezing anything...')
        freezing_params_during_warmup = list()
    elif freezing_params_during_warmup == 'model':
        freezing_params_during_warmup = list(model.parameters())
        print('freezing all model parameters for warmup ...', len(freezing_params_during_warmup))
    else:
        print('freezing given model parameters, total: ', len(freezing_params_during_warmup))

    param_groups = [{'params': model.parameters(), 'lr': float(train_cfg.lr) * 1},
                    {'params': criterion.proxies, 'lr': float(train_cfg.lr) * loss_cfg.proxy_lr_scale},
                    {'params': classifier.parameters()}]

    opt = prepare_optimizer(train_cfg, param_groups)
    scheduler = prepare_lr_scheduler(train_cfg, opt)

    dl_tr = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=train_cfg.train_batch_size,
                                        num_workers=run_cfg.cpu_workers, pin_memory=True)

    for epoch in range(1, train_cfg.train_epoch + 1):
        loss_avg = LossAverage()
        loss1_avg = LossAverage()
        loss2_avg = LossAverage()

        for param in freezing_params_during_warmup:
            param.requires_grad = (epoch > train_cfg.warmup_epoch)

        pbar = tqdm(enumerate(dl_tr), total=len(dl_tr), disable=run_cfg.tqdm_disable)
        for batch_idx, (x, y) in pbar:
            easydl.all_to_device(classifier, criterion, model, device=run_cfg.device)
            x, y = easydl.all_to_device(x, y, device=run_cfg.device)

            model.train()

            assert torch.isnan(x).sum() == 0
            o = model(x)
            loss1 = criterion(o, y)
            pred = classifier(o)
            loss2 = clf_loss(pred, y)
            loss = loss1 + loss2

            opt.zero_grad()
            loss.backward()

            assert torch.isnan(loss).sum() == 0
            for pg in opt.param_groups:
                torch.nn.utils.clip_grad_value_(pg['params'], train_cfg.clip_gradient)

            # must do step
            opt.step()
            loss1_avg.append(loss1)
            loss2_avg.append(loss2)
            loss_avg.append(loss)

            pbar.set_postfix({'epoch': epoch, 'batch': batch_idx, 'total_batch': len(dl_tr),
                              'loss_mean': loss_avg.mean()},
                             refresh=True)
        metric_logger.log({'pal_loss_mavg': loss1_avg.mean(), 'clf_loss_mavg': loss2_avg.mean()})
        on_epoch_end(opt, scheduler, run_cfg, metric_logger, loss_avg=loss_avg, pbar=pbar)

        if epoch_end_hook is not None:
            epoch_end_hook(locals=locals())