from easydl.utils import save_model
import easydl
import numpy as np

class LossAverage(object):

    def __init__(self):
        self.losses = []

    def append(self, loss):
        self.losses.append(loss)


class TrainAccuracyAverage():
    def __init__(self):
        self.prediction = []
        self.truth = []
        self.scores = []

    def update(self, pred, truth):
        """
        pred: torch.Tensor N*M, N*1, N,
        truth: torch.Tensor  N
        """
        truth = easydl.to_numpy(truth)
        assert truth.ndim == 1
        pred = np.squeeze(easydl.to_numpy(pred))
        assert pred.ndim == 2 or pred.ndim == 1
        self.truth.extend(truth)
        if pred.ndim == 2:
            pred = pred.argmax(axis=1)
        self.prediction.extend(pred)

    def accuracy(self):
        pred = np.array(self.prediction)
        truth = np.array(self.truth)
        return (pred == truth).mean()


class ModelSaveEpochHook:
    def __init__(self, model, filepath):
        self.model = model
        self.filepath = filepath

    def __call__(self, *args, **kwargs):
        save_model(self.model, self.filepath)



