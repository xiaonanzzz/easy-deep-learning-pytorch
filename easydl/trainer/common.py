from easydl.utils import save_model

class LossAverage(object):

    def __init__(self):
        self.losses = []

    def append(self, loss):
        self.losses.append(loss)


class ModelSaveEpochHook:
    def __init__(self, model, filepath):
        self.model = model
        self.filepath = filepath

    def __call__(self, *args, **kwargs):
        save_model(self.model, self.filepath)

