import os
from collections import Counter


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
    def __init__(self, *args, project='<noname>', tags=None, api_key='', prepare=True, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)
        self.project = project
        self.tags = tags or list()
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


def prepare_logger(wandb_key=None, filepath=None, project_name=None, tags=None, ):
    loggers = [PrintMetricLogger(filepath=filepath)]
    if wandb_key:
        assert project_name is not None
        loggers.append(WandbLogger(project=project_name, api_key=wandb_key, tags=tags))
    return MultiMetricLogger(loggers)