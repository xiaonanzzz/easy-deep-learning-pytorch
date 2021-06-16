import pandas as pd
import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.models import GoogLeNet
from torchvision.transforms import ToTensor, Normalize, Compose
from easydl.trainer.image_classification import ImageClassificationTrainer
from torch import nn
from easydl.evaluator.classification import ClassificationModelEvaluator

class SimpleNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(SimpleNet, self).__init__()
        self.channels = 64
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, ),
            nn.Conv2d(self.channels, self.channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)     # batch, channels, 1, 1
        x = self.classifier(x)
        if self.training:
            return x
        else:
            return torch.argmax(x, dim=1)

def main():
    import os
    print('working directory', os.getcwd())

    train_data = CIFAR10('tmp/data', train=True, download=True, transform=Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])]))
    print('data shape', train_data[0][0].shape, train_data[0])
    model = SimpleNet(10)

    trainer = ImageClassificationTrainer()

    def epoch_end_hook(**kwargs):
        model.eval()
        eval = ClassificationModelEvaluator(CIFAR10('tmp/data', train=False, download=True,
                                             transform=Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])])))
        eval.evaluate(model)
        model.train()
        print(kwargs)
        print('accuracy', eval.accuracy)

    trainer.train(model, train_data, epoch_end_hook=epoch_end_hook)

if __name__ == '__main__':
    main()

