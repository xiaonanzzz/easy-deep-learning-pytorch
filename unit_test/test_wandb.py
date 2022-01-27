from easydl.utils import WandbExperiment


def test_wandb():
    exp = WandbExperiment('debug', working_dir='~/tmp')
    print(exp.__dict__)

if __name__ == '__main__':
    test_wandb()