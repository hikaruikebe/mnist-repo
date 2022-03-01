# mnist-repo
handwritten digit predictor

Model is trained on MNIST dataset. BNN (Binarized Neural Network) uses binarized convolution layers, and CNN uses full-precision values to calculate.

Tester can test custom images of digits (0-9). Input image should be a square, i.e. same width and length. All .jpg images in directory ./images will be tested and a final score will be printed.