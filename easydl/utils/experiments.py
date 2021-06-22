import os


class ExpFolderLogger(object):
    def __init__(self):
        super(ExpFolderLogger, self).__init__()
        self.exp_dir = ''
        self.log_fname = 'exp-log.txt'
        self.log_path = None
        self.log_fobj = None

    def prepare_folder_logger(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        if self.log_fname is not None:
            self.log_path = os.path.join(self.exp_dir, self.log_fname)

        if self.log_fobj is None:
            self.log_fobj = open(self.log_path, mode='a')

    def file_path(self, fname):
        return os.path.join(self.exp_dir, fname)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, file=self.log_fobj, **kwargs)