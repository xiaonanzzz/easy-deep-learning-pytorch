import os
from collections import Counter
from easydl.config import get_config_from_cmd, RuntimeConfig, expand_path
import easydl


class MetricLogger(object):
    TrainLossMovingAverage = 'train_loss_mavg'
    TrainAccuracyMovingAverage = 'train_acc_mavg'
    TestAccuracy = 'test_accuracy'
    TrainAccuracy = 'train_accuracy'
    LastLr = 'last_lr'

    def __init__(self, run_cfg:RuntimeConfig, *args, **kwargs):

        self.run = None     # if None, wandb is not init or not used
        if run_cfg.debug:
            print('!!! debug mode, wandb logger is not used.')
            return
        elif run_cfg.use_wandb:
            self.log_count = Counter()
            self._init_wandb(run_cfg)
        else:
            print('Wandb is disabled...')

    def _init_wandb(self, run_cfg:RuntimeConfig):
        wandb_dir = expand_path(run_cfg.wandb_dir)
        os.makedirs(wandb_dir, exist_ok=True)
        tags = run_cfg.tags
        tags.append('easydl-v-' + easydl.__version__)
        key = get_wandb_key()
        if key is None:
            raise RuntimeError(
                'cannot find wandb key, Api key should be stored in ~/wandb_key.txt, or set by --wandb_key')
        import wandb
        wandb.login(key=key.strip())
        self.run = wandb.init(project=run_cfg.project_name, tags=tags, dir=wandb_dir)


    def update_config(self, config_dict):
        print('updating config...', config_dict)
        if self.run:
            self.run.config.update(config_dict)
            return

    def log(self, metric_dict, step=None):
        """

        :param metric_dict:
        :param step:            if all metrics are continous, then don't pass step, it will increase automatically
        :return:
        """
        if self.run :
            self.log_count.update(list(metric_dict.keys()))
            # if step is given, use it. otherwise, use the max step of metrics
            self.run.log(metric_dict, step=step or max(self.log_count.values()))

    def close(self):
        if self.run:
            self.run.finish()


def get_wandb_key(arg_name='wandb_key'):
    wandb_key = get_config_from_cmd(arg_name, '')
    if wandb_key != '':
        return wandb_key

    print('trying to get wandb key from os env')
    wandb_key = os.getenv('WANDB_KEY')
    if wandb_key is not None:
        return wandb_key
    print('trying to get wandb key from local ~/wandb_key.txt')
    if os.path.exists(os.path.expanduser('~/wandb_key.txt')):
        with open(os.path.expanduser('~/wandb_key.txt'), 'rt') as f:
            wandb_key = f.read()
            if wandb_key is not None:
                return wandb_key.strip()

    return None

