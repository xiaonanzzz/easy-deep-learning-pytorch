

class LossAverage(object):

    def __init__(self):
        self.losses = []

    def append(self, loss):
        self.losses.append(loss)