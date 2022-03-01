# mnist-repo
handwritten digit predictor

Model is trained on MNIST dataset using CPU (MacBook Pro). BNN (Binarized Neural Network) uses binarized convolution layers, and CNN uses full-precision values to calculate.

Tester can test custom images of digits (0-9). Input image should be a square, i.e. same width and length. All .jpg images in directory ./images will be tested and a final score will be printed.

CNN Scores:
    train: 97.1%
    test: 99.0%

Tester Score:
    7/10