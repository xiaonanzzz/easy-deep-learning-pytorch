import os
from collections import Counter
from easydl.config import get_config_from_cmd, RuntimeConfig, expand_path
import easydl

class PrinterInterface(object):

    def __call__(self, *args, **kwargs):
        """ exact same interface with bulti in print() function"""
        pass


class Printers(PrinterInterface):
    def __init__(self, printers=None, console=True, filepath=None):
        self.printers = printers or list()
        if console:
            self.printers.append(print)
        if filepath is not None:
            self.printers.append(FilePrinter(filepath))

    def __call__(self, *args, **kwargs):
        for printer in self.printers:
            printer(*args, **kwargs)


class FilePrinter(PrinterInterface):
    def __init__(self, filepath, prepare=True, mode='a+t'):
        self.filepath = os.path.expandvars(os.path.expanduser(filepath))
        self.file_obj = None
        self.mode = mode
        if prepare:
            self.prepare()

    def prepare(self):
        if self.file_obj is not None:
            return
        d = os.path.dirname(self.filepath)
        os.makedirs(d, exist_ok=True)
        self.file_obj = open(self.filepath, mode=self.mode)

    def __call__(self, *args, flush=True, **kwargs):
        if self.file_obj is None:
            import warnings
            warnings.warn('the file printer is not preapred')
            return
        print(*args, file=self.file_obj, flush=flush, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.file_obj.close()
        except:
            pass

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


class MultiMetricLogger(MetricLogger):
    def __init__(self, loggers, *args, **kwargs):
        super(MetricLogger, self).__init__(*args, **kwargs)
        self.loggers = list(loggers)

        print('multiple loggers', *loggers)

    def prepare(self):
        for x in self.loggers:
            x.prepare()

    def update_config(self, config_dict):
        for x in self.loggers:
            x.update_config(config_dict)

    def log(self, metric_dict, step=None):
        for x in self.loggers:
            x.log(metric_dict, step=step)

    def close(self):
        for x in self.loggers:
            x.close()


class PrintMetricLogger(MetricLogger):

    def __init__(self, *args, filepath=None, **kwargs):
        super(PrintMetricLogger).__init__(*args, **kwargs)
        self.config = {}
        self.log_count = Counter()
        self.print = Printers(filepath=filepath)

    def update_config(self, config_dict):
        self.config.update(config_dict)
        self.print(self.config)

    def log(self, metric_dict, step=None):
        self.log_count.update(list(metric_dict.keys()))
        step = step or max(self.log_count.values())
        self.print('step: {}, metrics: {}'.format(step, metric_dict))


class WandbLogger(MetricLogger):
    def __init__(self, *args, project='<noname>', tags=None, api_key='', working_dir=None, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)
        self.project = project
        self.tags = tags or list()
        self.api_key = api_key
        self.run = None
        self.working_dir = working_dir
        self.log_count = Counter()

        import wandb
        os.system('wandb login {}'.format(self.api_key))
        self.run = wandb.init(project=self.project, tags=self.tags, dir=self.working_dir)

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
                return wandb_key

    return None


def prepare_logger(wandb_key=None, filepath=None, project_name=None, tags=None, ):
    """
    wandb_key: [None, str (key), 'auto'], if 'auto', will try to get key from commend arg, system env, or ~/wandb_key.txt
    """
    loggers = [PrintMetricLogger(filepath=filepath)]

    if wandb_key == 'auto':
        wandb_key = get_wandb_key()

    if wandb_key is not None:
        assert project_name is not None
        wandb_logger = WandbLogger(project=project_name, api_key=wandb_key, tags=tags)
        loggers.append(wandb_logger)

    return MultiMetricLogger(loggers)


class WandbExperiment:
    def __init__(self, run_cfg: RuntimeConfig):
        """
        Api key should be stored in ~/wandb_key.txt, or set by --wandb_key
        """
        working_dir = expand_path(run_cfg.wandb_dir)
        os.makedirs(working_dir, exist_ok=True)
        tags = run_cfg.tags
        tags.append('easydl-v-' + easydl.__version__)
        key = get_wandb_key()
        if key is None:
            raise RuntimeError('cannot find wandb key, Api key should be stored in ~/wandb_key.txt, or set by --wandb_key')

        loggers = [PrintMetricLogger()]
        wandb_logger = WandbLogger(project=run_cfg.project_name, api_key=key,
                                         tags=tags,
                                         working_dir=working_dir)
        loggers.append(wandb_logger)
        self.metric_logger = MultiMetricLogger(loggers)


