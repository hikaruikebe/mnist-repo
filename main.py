import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

# load data
train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)

# access data in training set
sample = next(iter(train_set))
image, label = sample
# plt.imshow(image.squeeze(), cmap="gray")
# plt.show()


model = CNN_Network()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

print(model.parameters)
pytorch_total_params = sum(p.numel() for p in model.parameters())
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) #only trainable parameters

print("Number of Parameters: {:d}".format(pytorch_total_params))


for epoch in range(10):

    total_loss = 0
    total_correct = 0

    for batch in train_loader:  # Get Batch
        images, labels = batch

        preds = model(images)  # Pass Batch
        loss = F.cross_entropy(preds, labels)  # Calculate Loss

        optimizer.zero_grad()
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    state_dict = model.state_dict()
    # print(state_dict)
    torch.save(state_dict, "our_model.tar")

    print("epoch", epoch, "acc:", total_correct / len(train_set), "loss:", total_loss)


with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(model, prediction_loader)

preds_correct = get_num_correct(train_preds, train_set.targets)
print("total correct:", preds_correct)
print("accuracy:", preds_correct / len(train_set))

num_correct = get_num_correct(preds, labels)

cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
# print(type(cm))
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, train_set.classes)
