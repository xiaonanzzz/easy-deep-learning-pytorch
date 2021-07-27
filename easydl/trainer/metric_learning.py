import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data.sampler import BatchSampler

from easydl.evaluator import MetricLearningModelEvaluatorSingleSet
from easydl.utils import l2_norm, binarize
from easydl.trainer.optimizer import OptimizerArgs, prepare_optimizer
from easydl.trainer.lr_scheduler import prepare_lr_scheduler, LRSchedulerArgs
from easydl.trainer import EpochTrainer
from easydl.config import TqdmConfig
from easydl.utils.experiments import MetricLogger, PrintMetricLogger

from tqdm import *

class ProxyAnchorLoss(torch.nn.Module):
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

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


class ProxyAnchorLossEmbeddingModelTrainer(EpochTrainer, OptimizerArgs, LRSchedulerArgs, TqdmConfig):
    def __init__(self, model, train_dataset, sz_embedding, nb_classes, *args,
                 optimizer='adamw', lr=1e-4, weight_decay=1e-4, momentum=0.9,      # default setting for proxy anchor loss
                 metric_logger=PrintMetricLogger(),
                 **kwargs):
        super(ProxyAnchorLossEmbeddingModelTrainer, self).__init__(*args,
                optimizer=optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum, **kwargs)
        """
        some of the default parameters are used according to the paper: 
        Proxy Anchor Loss for Deep Metric Learning https://arxiv.org/abs/2003.13911
        """
        self.mrg = 0.1
        self.alpha = 32
        self.lr_decay_step = 10
        self.lr_decay_gamma = 0.5
        self.nb_epochs = 60
        self.nb_workers = 8    # number of cpus process for loading data
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proxy_lr_scale = 100

        # run time arguments
        self.model = model           # take input from train_dataset[i][0], generate embedding vector of D
        self.train_dataset = train_dataset   # train_dataset[i][0] is input data, train_dataset[i][1] is label int
        self.sz_embedding = sz_embedding    # number of dimensions in embedding D
        self.nb_classes = nb_classes      # number of classes in set( train_dataset[i][1] )
        self.metric_logger = metric_logger       # by default metric logger, do nothing

    def train(self):
        args = self
        model = self.model
        train_dataset = self.train_dataset

        criterion = ProxyAnchorLoss(nb_classes=self.nb_classes, sz_embed=self.sz_embedding, mrg=self.mrg,
                                        alpha=self.alpha)

        param_groups = [{'params': model.parameters(), 'lr': float(args.lr) * 1},
                        {'params': criterion.proxies, 'lr': float(args.lr) * self.proxy_lr_scale}]

        opt = prepare_optimizer(args, param_groups)

        scheduler = prepare_lr_scheduler(args, opt)

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
            pbar = tqdm(enumerate(dl_tr), total=len(dl_tr), disable=self.tqdm_disable)

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
                torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

                losses_per_epoch.append(loss.data.cpu().numpy())
                opt.step()

                loss_mean = np.mean(losses_per_epoch)
                pbar.set_description( 'Train Epoch: {} Loss: {:.6f}'.format(epoch, loss_mean), refresh=False)

            self.metric_logger.log({'training_loss': loss_mean})
            pbar.close()
            scheduler.step()

            if self.epoch_end_hook is not None:
                epoch_end_hook = self.epoch_end_hook
                epoch_end_hook(locals=locals())


class EpochEndEvaluationHook(TqdmConfig):
    def __init__(self, model, testds, *args, metric_logger=PrintMetricLogger(), **kwargs):
        """

        :param model:           pytorch embedder model x -> x'
        :param testds:          testing dataset,  testds[i] -> x, y
        """
        super(EpochEndEvaluationHook, self).__init__(*args, **kwargs)
        self.model = model
        self.testds = testds
        self.recall_at_k_list = []      # each one is a dictionary {'k': value}
        self.metric_logger = metric_logger    # by default, do nothing.

    def __call__(self, *args, **kwargs):
        model = self.model
        model.eval()
        eval = MetricLearningModelEvaluatorSingleSet(self.testds)
        eval.evaluate(model)
        model.train()
        self.recall_at_k_list.append(eval.recall_at_k)

        self.metric_logger.log({'recall@{}'.format(k): v for k,v in self.recall_at_k_list[-1].items() })
