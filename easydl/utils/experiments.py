import os
from collections import Counter

class ExpFolderLogger(object):
    def __init__(self):
        super(ExpFolderLogger, self).__init__()
        self.exp_dir = ''
        self.log_fname = 'exp-log.txt'
        self.log_path = None
        self.log_fobj = None
        self.redirect_print = True

    @property
    def logfile(self):
        return self.log_fobj

    def prepare_folder_logger(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        if self.log_fname is not None:
            self.log_path = os.path.join(self.exp_dir, self.log_fname)

        if self.log_fobj is None:
            self.log_fobj = open(self.log_path, mode='a')

    def file_path(self, fname):
        return os.path.join(self.exp_dir, fname)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.redirect_print and self.log_fobj is not None:
            print(*args, file=self.log_fobj, **kwargs)


class MetricLogger(object):
    def __init__(self, *args, **kwargs):
        super(MetricLogger, self).__init__(*args, **kwargs)

    def prepare(self):
        pass

    def update_config(self, config_dict):
        pass

    def log(self, metric_dict, step=None):
        pass

    def close(self):
        pass

class PrintMetricLogger(MetricLogger):

    def __init__(self, *args, **kwargs):
        super(PrintMetricLogger).__init__(*args, **kwargs)
        self.config = {}
        self.log_count = Counter()

    def update_config(self, config_dict):
        self.config.update(config_dict)

    def log(self, metric_dict, step=None):
        self.log_count.update(list(metric_dict.keys()))
        step = step or max(self.log_count.values())
        print('step: {}, metrics: {}'.format(step, metric_dict))


class WandbLogger(MetricLogger):
    def __init__(self, *args, project='<noname>', tags=[], api_key='', prepare=True, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)
        self.project = project
        self.tags = tags
        self.api_key = api_key
        self.run = None
        self.log_count = Counter()

        if prepare:
            self.prepare()

    def prepare(self):
        import wandb
        os.system('wandb login {}'.format(self.api_key))
        self.run = wandb.init(project=self.project, tags=self.tags)

    def update_config(self, config_dict):
        if self.run is None:
            return
        self.run.config.update(config_dict)

    def log(self, metric_dict, step=None):
        """

        :param metric_dict:
        :param step:            if all metrics are continous, then don't pass step, it will increase automatically
        :return:
        """
        if self.run is None:
            return
        self.log_count.update(list(metric_dict.keys()))
        # if step is given, use it. otherwise, use the max step of metrics
        self.run.log(metric_dict, step=step or max(self.log_count.values()))

    def close(self):
        if self.run is None:
            return
        self.run.finish()

