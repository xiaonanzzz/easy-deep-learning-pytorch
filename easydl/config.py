import torch


class DeviceConfig():
    def __init__(self):
        super(DeviceConfig, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
