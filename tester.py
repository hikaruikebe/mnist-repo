import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import seaborn as sns
import os

from bnn import *
from cnn import *

path_model = "our_model.tar"
model = CNN_Network()
model.load_state_dict(torch.load(path_model))
model.eval()

directory = "./images"

sum = 0
total = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        label = int(filename[0])
        path_img = os.path.join(directory, filename)
        img = Image.open(path_img)
        png_directory = os.path.join(directory, filename[0] + ".png")
        img.save(png_directory)

        img_gray = cv2.imread(png_directory, cv2.IMREAD_GRAYSCALE)
        (thresh, img) = cv2.threshold(
            img_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        f, ax = plt.subplots(figsize=(16, 16))
        # sns.heatmap(img, annot=True, fmt=".1f", square=True, cmap="YlGnBu")
        # plt.show()

        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32)
        img /= 255.0
        img = torch.Tensor(img).unsqueeze(axis=0).unsqueeze(axis=0)

        pred = model(img)
        _, predicted = torch.max(pred.data, 1)
        outcome = "Not correct"
        total += 1
        if predicted == label:
            sum += 1
            outcome = "Correct"

        print(
            "True label: ",
            label,
            "Predicted label: ",
            predicted.item(),
            "Outcome: ",
            outcome,
        )
        continue
    else:
        continue

print("Correct predictions: ", sum, "/", total)
