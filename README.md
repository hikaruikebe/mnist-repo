# mnist-repo
handwritten digit predictor

Model is trained on MNIST dataset using CPU (MacBook Pro). BNN (Binarized Neural Network) uses binarized convolution layers, and CNN uses full-precision values to calculate.

CNN model parameters can be accessed from cnn_model.tar. tester.py will automatically load from cnn_model.tar and create a cnn model.

Tester can test custom images of digits (0-9). Input image should be a square, i.e. same width and length. All .jpg images in directory ./images will be tested and a final score will be printed.

CNN Scores:
    train: 97.1%
    test: 99.0%

Tester Score:
    9/10

To test your own digits:
1. Take an image of a [digit] and crop it so that the image shape is a square
2. Save as [digit].jpg
3. Upload to images/ file
4. Run tester.py
    a. tester.py will use latest trained parameters for cnn.py

References:
Binary Neural Network (BNN):
https://github.com/itayhubara/BinaryNet.pytorch
https://arxiv.org/pdf/1602.02830.pdf

MNIST:
http://yann.lecun.com/exdb/mnist/

Other CNN materials:
https://colab.research.google.com/github/chokkan/deeplearningclass/blob/master/mnist.ipynb?authuser=2#scrollTo=z7HHJrQXibt0
https://deeplizard.com/learn/playlist/PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG
