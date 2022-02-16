import os.path

from easydl.experiments import WandbExperiment


def test_wandb():
    exp = WandbExperiment('debug', working_dir='~/tmp')
    print(exp.__dict__)

    assert os.path.exists(exp.working_dir)

if __name__ == '__main__':
    test_wandb()