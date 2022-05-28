import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from modulefinder import Module
from easydl.algorithm.metric_learning import *
from easydl.algorithm.losses import *
from easydl.config import RuntimeConfig, TrainingConfig, get_config_from_cmd
from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.experiments import MetricLogger
from easydl.image_transform import resnet_transform_test, resnet_transform_train, timm_image_transform_imagenet_default
from easydl.models.image_model import Resnet50PALVersion
from easydl.models.mlp_model import EmbedderClassifier, LinearEmbedder, L2Normalization
from easydl.models.convnext import convnext_base, get_convnext_version_augmentation_config
import os


class ProxyAnchorLossSoft(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(F.normalize(X), F.normalize(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        P_one_hot = P_one_hot + 0.1/self.nb_classes
        P_one_hot = P_one_hot / (P_one_hot.sum(dim=1, keepdim=True))
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        P_sim_sum = (pos_exp * P_one_hot).sum(dim=0)
        N_sim_sum = (neg_exp * N_one_hot).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).mean()
        neg_term = torch.log(1 + N_sim_sum).mean()
        loss = pos_term + neg_term

        return loss


def train_model(model, train_ds, nb_classes, metric_logger,
                                                    train_cfg: TrainingConfig, run_cfg: RuntimeConfig,
                                                    loss_cfg: ProxyAnchorLossConfig,
                                                    epoch_end_hook=None, freezing_params_during_warmup=None):
    """
    freezing_params_during_warmup:  None: no freezing, "model": freezing all model parameters, list(params): freezing the given list of params
    """
    print('configurations', train_cfg, loss_cfg)
    criterion = ProxyAnchorLossSoft(nb_classes=nb_classes, sz_embed=loss_cfg.embedding_size, mrg=loss_cfg.margin, alpha=loss_cfg.alpha)

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

def convnext_exp():

    # prepare configurations
    train_cfg = TrainingConfig(optimizer='adamw', lr=1e-4, weight_decay=1e-4, lr_scheduler_type='step',
                               lr_decay_step=5, train_batch_size=120, train_epoch=60, warmup_epoch=1)
    train_cfg.pretrained = True
    train_cfg.from_other(get_convnext_version_augmentation_config())
    train_cfg.update_values_from_cmd()

    run_cfg = RuntimeConfig(project_name='convnext_cub')
    run_cfg.update_values_from_cmd()
    
    algo_cfg = ProxyAnchorLossConfig(embedding_size=1024)
    algo_cfg.update_values_from_cmd()

    # prepare experiments
    cub_exp = CubMetricLearningExperiment()

    metric_logger = MetricLogger(run_cfg)

    # prepare model
    # because it's pre-trained on 22k, so, set num_classes = 21841
    model = convnext_base(pretrained=train_cfg.pretrained, in_22k=True, num_classes=21841) 
    # replace the head with a linear embedder
    model.head = L2Normalization()
    freezing_params = list(set(model.parameters()) - set(model.head.parameters()))
    
    cub_exp.train_ds.change_image_transform(timm_image_transform_imagenet_default(train_cfg))
    cub_exp.test_ds.change_image_transform(resnet_transform_test)

    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())

    def epoch_end(**kwargs):
        print('evaluting the model on testing data...')
        recall_at_k = cub_exp.evaluate_model(model)
        metric_logger.log({'recall@{}'.format(k): v for k, v in recall_at_k.items()})

        # save model 
        local_path = os.path.join(metric_logger.local_run_path, 'last_model.torch')
        torch.save(model.state_dict(), local_path)
        if metric_logger.get_best_step('recall@1') == metric_logger.current_step:
            metric_logger.set_summary('best/step', metric_logger.current_step)
            metric_logger.set_summary('best/recall@1', metric_logger.metrics['recall@1'][metric_logger.current_step])

            best_path = os.path.join(metric_logger.local_run_path, 'best_model.torch')
            torch.save(model.state_dict(), best_path)


    # run experiment
    train_model(model, cub_exp.train_ds, cub_exp.train_classes,
                                                    metric_logger, train_cfg, run_cfg, algo_cfg,
                                                                    epoch_end_hook=epoch_end,
                                                                    freezing_params_during_warmup=freezing_params)

if __name__ == '__main__':
    convnext_exp()

