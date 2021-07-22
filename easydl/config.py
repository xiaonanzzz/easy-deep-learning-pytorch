import torch


class DeviceConfig(object):
    def __init__(self, *args, **kwargs):
        super(DeviceConfig, self).__init__(*args, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TqdmConfig(object):
    def __init__(self, *args, tqdm_disable=True, **kwargs):
        super(TqdmConfig, self).__init__(*args, **kwargs)
        self.tqdm_disable = tqdm_disable