import torch
import torch.nn as nn
import torch.nn.functional as F
from binarized import *


class BNN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        # self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)

        self.fc1 = BinarizeLinear(in_features=1 * 28 * 28, out_features=2048)
        self.fc2 = BinarizeLinear(in_features=2048, out_features=2048)
        self.fc3 = BinarizeLinear(in_features=2048, out_features=2048)
        self.fc4 = BinarizeLinear(in_features=2048, out_features=10)

        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.htanh1 = nn.Hardtanh()
        self.htanh2 = nn.Hardtanh()
        self.htanh3 = nn.Hardtanh()

        self.drop = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2)

        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2)

        # x = x.reshape(-1, 1 * 28 * 28)
        # x = self.fc1(x)
        # x = F.relu(x)

        # x = self.fc2(x)
        # x = F.relu(x)

        # x = self.out(x)

        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        # x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        x = self.logsoftmax(x)
        return x


def bnn(**kwargs):
    # num_classes = kwargs.get( 'num_classes', 1000)
    return BNN_Network()
