import unittest
import easydl
import torch


class MyTestCase(unittest.TestCase):

    def test_prepare_opt_lr(self):
        from easydl.training_common import prepare_lr_scheduler, prepare_optimizer
        from easydl.mlp_model import MLPEmbedder
        config = easydl.TrainingConfig(lr_scheduler_type='cosine')
        model = MLPEmbedder([128, 64])

        param_groups = [model.parameters()]
        opt = prepare_optimizer(config, param_groups)
        lr = prepare_lr_scheduler(config, opt)
        print(opt, lr)
        assert lr.get_last_lr()[0] == config.lr, (lr.get_last_lr(), config.lr)


if __name__ == '__main__':
    unittest.main()

