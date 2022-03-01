# adapted from https://colab.research.google.com/github/chokkan/deeplearningclass/blob/master/mnist.ipynb?authuser=2#scrollTo=v8H-xg9pom28
# adapted from https://deeplizard.com/learn/playlist/PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# from resources.plotcm import plot_confusion_matrix

from plotcm import plot_confusion_matrix
from utils import *


# from plotcm import plot_confusion_matrix

import pdb
import os
import numpy as np
import cv2
from PIL import Image
from bnn import BNN_Network
from cnn import CNN_Network

torch.set_printoptions(linewidth=120)

# get data or download if necessary
train_set = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

test_set = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

# load data
train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)

# access data in training set
# sample = next(iter(train_set))
# image, label = sample
# plt.imshow(image.squeeze(), cmap="gray")
# plt.show()


model = CNN_Network()
# optimizer = optim.Adam(model.parameters(), lr=0.001) # deep lizard
# optimizer = optim.SGD(model.parameters(), lr=0.01) # hubara bnn
optimizer = optim.SGD(model.parameters(), lr=0.001)  # google colab deep nn tutorial

loss_function = nn.CrossEntropyLoss(size_average=False)

print(model.parameters)
pytorch_total_params = sum(p.numel() for p in model.parameters())
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) #only trainable parameters

print("Number of Parameters: {:d}".format(pytorch_total_params))

epochs = 100
for epoch in range(epochs):

    # train
    model.train()
    total_loss = 0
    total_correct = 0

    for i, batch in enumerate(train_loader):  # Get Batch
        images, labels = batch

        preds = model(images)  # Pass Batch

        # total_correct += get_num_correct(preds, labels)
        _, predicted = torch.max(preds.data, 1)
        total_correct += (predicted == labels).sum().item()

        loss = loss_function(preds, labels)  # Calculate Loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        # if i % 10 == 0:
        #     print(
        #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #             epoch,
        #             i * len(images),
        #             len(train_loader.dataset),
        #             100.0 * i / len(train_loader),
        #             loss.item(),
        #         )
        #     )

    state_dict = model.state_dict()
    # print(state_dict)
    torch.save(state_dict, "cnn_model.tar")

    # test
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for i, (preds, labels) in enumerate(test_loader):
            # preds, labels = Variable(preds), Variable(labels)
            output = model(preds)
            test_loss += loss_function(output, labels).item()  # sum up batch loss
            # pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "Epoch: {:d}\tTest set: Average loss: {:.4f}\tAccuracy: {}/{} ({:.0f}%)".format(
            epoch,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    # print("epoch", epoch, "acc:", total_correct / len(train_set), "loss:", total_loss)


with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(model, prediction_loader)

# preds_correct = get_num_correct(train_preds, train_set.targets)
# print("total correct:", preds_correct)
# print("accuracy:", preds_correct / len(train_set))

# num_correct = get_num_correct(preds, labels)

cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
# print(type(cm))
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, train_set.classes)
