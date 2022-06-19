from cProfile import label
import numpy as np
import torch
from tqdm import tqdm

from easydl.config import RuntimeConfig
from easydl.config import TrainingConfig
from easydl.experiments import MetricLogger
from easydl.training_common import TrainAccuracyAverage, prepare_optimizer, prepare_lr_scheduler
from easydl.batch_processing import batch_process_x_y_dataset_and_concat

def evaluate_classification_model(model, test_dataset, runcfg:RuntimeConfig, **kwargs):
    model.eval()
    ypred, ytrue = batch_process_x_y_dataset_and_concat(test_dataset, model, batch_size=runcfg.infer_batch_size,
                                       save_index=1, tqdm_disable=False, tqdm_description='evaluation', **kwargs)
    assert ypred.shape[0] == ytrue.shape[0]
    if len(ypred.shape) == 2:
        ypred = torch.argmax(ypred, dim=1)
    assert ypred.shape == ytrue.shape
    accuracy = float(torch.mean((ypred == ytrue).double()))
    met_dict = {'accuracy': accuracy}
    return met_dict


def train_image_classification_model_2021_nov(model, train_ds, train_cfg: TrainingConfig, run_cfg: RuntimeConfig,
                                              metric_logger: MetricLogger,
                                              test_ds=None, epoch_end_hook=None, eval_train_ds=False):
    """
    epoch_end_hook will be called at the end of epoch, epoch_end_hook(locals=locals())
    """
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    param_groups = [{'params': model.parameters(), 'lr': float(train_cfg.lr) * 1}]

    opt = prepare_optimizer(train_cfg, param_groups)
    scheduler = prepare_lr_scheduler(train_cfg, opt)
    dl_tr = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        batch_size=train_cfg.train_batch_size,
        num_workers=run_cfg.cpu_workers,
        pin_memory=True
    )

    for epoch in range(1, train_cfg.train_epoch + 1):
        losses_per_epoch = []
        acc_avg = TrainAccuracyAverage()

        model.train()
        model.to(run_cfg.device)
        criterion.to(run_cfg.device)
        pbar = tqdm(enumerate(dl_tr), disable=run_cfg.tqdm_disable, total=len(dl_tr))
        for batch_idx, (x, y) in pbar:
            assert torch.isnan(x).sum() == 0

            m = model(x.to(run_cfg.device))
            loss = criterion(m, y.to(run_cfg.device))

            opt.zero_grad()
            loss.backward()

            assert torch.isnan(loss).sum() == 0
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            opt.step()

            # plot info [optional]
            acc_avg.update(m, y)
            losses_per_epoch.append(loss.data.cpu().numpy())
            pbar.set_postfix({'epoch': epoch,
                              'batch': batch_idx,
                              'numbatch': len(dl_tr),
                              'loss_mean': np.mean(losses_per_epoch),
                              'acc_mean': acc_avg.accuracy()}
                             , refresh=True)
        pbar.close()
        scheduler.step()

        # epoch end
        metric_logger.log({metric_logger.LastLr: float(scheduler.get_last_lr()[0]),
                           metric_logger.TrainLossMovingAverage: np.mean(losses_per_epoch),
                           MetricLogger.TrainAccuracyMovingAverage: acc_avg.accuracy()
                           })
        model.eval()
        if eval_train_ds:
            met = evaluate_classification_model(model, train_ds, run_cfg)
            metric_logger.log({metric_logger.TrainAccuracy: met['accuracy']})
        if test_ds is not None:
            met = evaluate_classification_model(model, test_ds, run_cfg)
            metric_logger.log({metric_logger.TestAccuracy: met['accuracy']})
        if epoch_end_hook is not None:
            print('epoch done. calling hook function...')
            epoch_end_hook(locals=locals())