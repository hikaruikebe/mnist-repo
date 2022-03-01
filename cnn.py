# https://colab.research.google.com/github/chokkan/deeplearningclass/blob/master/mnist.ipynb?authuser=2#scrollTo=v8H-xg9pom28

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        # Flatten(),
        self.linear1 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout()
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = x.view(-1, 512)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        return x


def cnn(**kwargs):
    # num_classes = kwargs.get( 'num_classes', 1000)
    return CNN_Network()
