import json
import warnings
import numpy as np
import os
from collections import Counter, defaultdict
from easydl.config import get_config_from_cmd, RuntimeConfig, expand_path, TrainingConfig
import easydl
import pandas as pd


class MetricLogger(object):
    """
    steps are all start from 1
    """

    TrainLossMovingAverage = 'train_loss_mavg'
    TrainAccuracyMovingAverage = 'train_acc_mavg'
    TestAccuracy = 'test_accuracy'
    TrainAccuracy = 'train_accuracy'
    LastLr = 'last_lr'

    def __init__(self, run_cfg:RuntimeConfig, *args, key_metric=None, **kwargs):

        self.run = None     # if None, wandb is not init or not used
        self.run_cfg = run_cfg
        self.config = {}
        self.summary = {}

        if run_cfg.debug:
            print('!!! debug mode, wandb logger is not used.')
            self.local_run_path = os.path.join(run_cfg.local_exp_dir, 'debug')
            os.makedirs(self.local_run_path, exist_ok=True)
            return
        elif run_cfg.use_wandb:
            self.log_count = Counter()
            self._init_wandb(run_cfg)
            self.local_run_path = os.path.join(run_cfg.local_exp_dir, run_cfg.project_name, self.run.name)
            os.makedirs(self.local_run_path, exist_ok=True)
        else:
            for i in range(1, 10000):
                local_path = os.path.join(run_cfg.local_exp_dir, run_cfg.project_name, 'run-{}'.format(i))
                if not os.path.exists(local_path):
                    break

            self.local_run_path = local_path
            os.makedirs(run_cfg.local_run_path, exist_ok=True)
            print('Wandb is disabled...')
        
        self.metrics = defaultdict(list)       # key -> [ metric numbers (float) ], 
        print('local dir for the run is', self.local_run_path)
        

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
            
        self.config.update(config_dict)

        save_path = os.path.join(self.local_run_path, 'config.json')
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=1)


    def log(self, metric_dict, step=None):
        """

        :param metric_dict:
        :param step:            if all metrics are continous, then don't pass step, it will increase automatically
        :return:
        """

        for k, v in metric_dict.items():
            self.metrics[k].append(v)

        if self.run :
            # if step is given, use it. otherwise, use the max step of metrics
            self.run.log(metric_dict, step=step or self.current_step)

        # save metrics to local file
        names = [k for k in self.metrics]
        dfs = [pd.DataFrame({k: self.metrics[k]}) for k in names]
        df = pd.concat(dfs, ignore_index=True, axis=1)
        df.columns = names

        save_path = os.path.join(self.local_run_path, 'metrics.csv')
        df.to_csv(save_path, index=False)

    def set_summary(self, key, value):
        self.summary[key] = value
        
        if self.run:
            self.run.summary[key] = value
        
        save_path = os.path.join(self.local_run_path, 'summary.json')
        with open(save_path, 'w') as f:
            json.dump(self.summary, f, indent=1)
        
        

    def get_best_step(self, key):
        return np.argmax(self.metrics[key])
        
    @property
    def current_step(self):
        # step starts from 0 for consistency
        return max(map(lambda x: len(x), self.metrics.values())) - 1

    def get_path(self, rel_path):
        return self.local_run_path


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

