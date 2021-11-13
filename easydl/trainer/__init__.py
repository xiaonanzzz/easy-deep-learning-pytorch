


class EpochTrainer(object):

    def __init__(self, *args, **kwargs):
        super(EpochTrainer, self).__init__(*args, **kwargs)
        self.epoch_end_hook = None  # optional

    def train(self):
        pass

    def on_epoch_end(self):
        if self.epoch_end_hook is not None:
            self.epoch_end_hook()

