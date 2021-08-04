import torch

class ConfigBase(object):
    BASIC_CLASSES = set([int, str, float])

    @staticmethod
    def is_core_config(value):
        if type(value) in ConfigBase.BASIC_CLASSES:
            return True
        if isinstance(value, list):
            # support a list of basic classes
            for x in value:
                if not type(x) in ConfigBase.BASIC_CLASSES:
                    return False
            return True
        return False

    def __init__(self, *args, **kwargs):
        super(ConfigBase, self).__init__(*args, **kwargs)

    def core_config(self):
        return {k: v for k, v in self.__dict__.items() if ConfigBase.is_core_config(v)}

    def all_config(self):
        return self.__dict__.copy()


class ConfigContainer(object):
    def __init__(self, *args, configures=None, **kwargs):
        super(ConfigContainer, self).__init__(*args, **kwargs)
        if configures is not None:
            for x in configures:
                self.__dict__[x.__class__.__name__] = x

    def __getitem__(self, item):
        if isinstance(item, str):
            cls, att = item.split('.')
            for x in self.__dict__.values():
                if x.__class__.__name__ == cls:
                    return getattr(x, att)
            return KeyError('{} not in the configures', item)
        else:
            for x in self.__dict__.values():
                if x.__class__ == item:
                    return x
            return KeyError('{} not in the configures', item)

    def check(self, list_of_classes):
        existing = [x.__class__ for x in self.__dict__.values()]
        for cls in list_of_classes:
            if cls not in existing:
                return ValueError('{} not in the configure manager'.format(cls))

    def all_config_dict(self):
        all_dict = {}
        for x in self.__dict__.values():
            d = x.all_config()
            for k, v in d.items():
                all_dict['{}.{}'.format(x.__class__.__name__, k)] = v
        return all_dict


class ConfigConsumer(object):

    def __init__(self, *args, configure_manager=None, **kwargs):
        super(ConfigConsumer, self).__init__(*args, **kwargs)
        self.configure_container: ConfigContainer = configure_manager

    def get_config(self, item):
        return self.configure_container[item]

    def check_config(self, list_of_classes):
        # should be called after init or any time you want to make sure configs are included
        self.configure_container.check(list_of_classes=list_of_classes)


class RuntimeConfig(ConfigBase):
    def __init__(self, *args, device=None, cpu_workers=2, tqdm_disable=True, infer_batch_size=32, **kwargs):
        super(RuntimeConfig, self).__init__(*args, **kwargs)
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu_workers = cpu_workers
        self.tqdm_disable = tqdm_disable
        self.infer_batch_size = infer_batch_size


class OptimizerConfig(ConfigBase):
    def __init__(self, *args, optimizer='sgd', lr=0.1, weight_decay=1e-4, momentum=0.9, **kwargs):
        super(OptimizerConfig, self).__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum


class LRSchedulerConfig(ConfigBase):
    def __init__(self, *args, lr_decay_step=10, lr_decay_gamma=0.5, **kwargs):
        super(LRSchedulerConfig, self).__init__(*args, **kwargs)
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma


class SklearnConfig(ConfigBase):
    def __init__(self, *args, num_jobs=2, **kwargs):
        super(SklearnConfig, self).__init__(*args, **kwargs)
        self.num_jobs = num_jobs


class CommonDeepLearningConfigContainer(ConfigContainer):
    def __init__(self, *args, **kwargs):
        super(CommonDeepLearningConfigContainer, self).__init__(*args, **kwargs)
        self.optimizer = OptimizerConfig()
        self.lr_scheduler = LRSchedulerConfig()
        self.runtime = RuntimeConfig()
