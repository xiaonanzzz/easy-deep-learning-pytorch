import torch
from easydl.utils import process_dataset_by_batch
from easydl.config import DeviceConfig

class ClassificationModelEvaluator(DeviceConfig):
    def __init__(self, test_dataset):
        super(ClassificationModelEvaluator, self).__init__()
        self.test_dataset = test_dataset
        self.accuracy = 0
        self.batch_size = 32
        self.ypred = None
        self.ytrue = None

    def evaluate(self, model):
        out, ys = process_dataset_by_batch(self.test_dataset, model, batch_size=self.batch_size, save_index=1, device=self.device)

        ypred = torch.cat(out, dim=0)
        ytrue = torch.cat(ys, dim=0)
        print(ypred.shape, ytrue.shape)

        self.accuracy = torch.mean((ypred == ytrue).double())
        self.ypred = ypred
        self.ytrue = ytrue