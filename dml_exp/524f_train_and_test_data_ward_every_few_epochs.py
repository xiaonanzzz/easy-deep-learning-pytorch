import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sklearn.cluster import AgglomerativeClustering

from easydl.batch_processing import batch_process_x_y_dataset_and_concat
from easydl.datasets.image_dataset import ImageDataset



from modulefinder import Module
from easydl.algorithm.metric_learning import *
from easydl.config import RuntimeConfig, TrainingConfig, get_config_from_cmd
from easydl.datasets.cub import CubMetricLearningExperiment
from easydl.experiments import MetricLogger
from easydl.image_transform import resnet_transform_test, resnet_transform_train, timm_image_transform_imagenet_default
from easydl.models.image_model import Resnet50PALVersion
from easydl.models.mlp_model import EmbedderClassifier, LinearEmbedder, L2Normalization
from easydl.models.convnext import convnext_base, get_convnext_version_augmentation_config
import os

def proxy_anchor_loss_training(model, train_ds, nb_classes, metric_logger: MetricLogger,
                                                    train_cfg: TrainingConfig, run_cfg: RuntimeConfig,
                                                    loss_cfg: ProxyAnchorLossConfig, train_epoches=5,
                                                    epoch_end_hook=None, freezing_params_during_warmup=None):
    """
    freezing_params_during_warmup:  None: no freezing, "model": freezing all model parameters, list(params): freezing the given list of params
    """
    print('configurations', train_cfg, loss_cfg)
    criterion = ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=loss_cfg.embedding_size, mrg=loss_cfg.margin, alpha=loss_cfg.alpha) 

    param_groups = [{'params': model.parameters(), 'lr': float(train_cfg.lr) * 1},
                    {'params': criterion.proxies, 'lr': float(train_cfg.lr) * loss_cfg.proxy_lr_scale}]
    opt = prepare_optimizer(train_cfg, param_groups)
    scheduler = prepare_lr_scheduler(train_cfg, opt)

    for epoch_i in range(train_epoches):
        
        dl_tr = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=train_cfg.train_batch_size, num_workers=run_cfg.cpu_workers, pin_memory=True)
        loss_avg = LossAverage()

        pbar = tqdm(enumerate(dl_tr), total=len(dl_tr), disable=run_cfg.tqdm_disable)
        for batch_idx, (x, y) in pbar:

            loss = forward_backward_one_step(model, criterion, x, y, opt, train_cfg, run_cfg)

            loss_avg.append(loss)

            pbar.set_postfix({'epoch': metric_logger.current_step, 'batch': batch_idx, 'total_batch': len(dl_tr),
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
    train_cfg.ward_threshold = 1.5
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

    metric_logger.update_config(train_cfg.dict())
    metric_logger.update_config(algo_cfg.dict())

    train_img_transform = timm_image_transform_imagenet_default(train_cfg)
    def merge_train_test() -> ImageDataset:
        print('generating a mixed dataset...')
        model.to(run_cfg.device)
        model.eval()

        train_ds = cub_exp.get_train_ds()
        test_ds = cub_exp.get_test_ds(cub_exp.testing_transform)    # using default transform
        print('performing clustering on testing set to generate pseudo labels')
        x, y = batch_process_x_y_dataset_and_concat(test_ds, model, tqdm_disable=False)

        cls = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=train_cfg.ward_threshold)
        ypred = cls.fit_predict(x.numpy())
        print('# clusters', len(set(ypred)))

        impath = train_ds.im_paths
        labels = list(map(lambda x: f'train-{x}', train_ds.labels))

        impath.extend(test_ds.im_paths)
        labels.extend(map(lambda x: f'test-{int(x)}', ypred))
        print('last few labels', labels[-5])

        train_ds = ImageDataset(impath, labels, transform=train_img_transform, verbose=True)
        print('total number of classes', train_ds.num_labels)
        return train_ds

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
    for i in range(10):
        print(f'training for big epoch {i}')
        train_ds = merge_train_test()
        proxy_anchor_loss_training(model, train_ds, train_ds.num_labels,
                                                    metric_logger, train_cfg, run_cfg, algo_cfg,
                                                                    epoch_end_hook=epoch_end,
                                                                    freezing_params_during_warmup=freezing_params)

if __name__ == '__main__':
    convnext_exp()

