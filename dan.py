
from EasyNN.model import Network, ReLU, LogSoftMax
from EasyNN.dataset.mnist.number import dataset, labels
from EasyNN.utilities.image.image import Image
from EasyNN.utilities.image.mnist.show import show
from EasyNN.utilities.parameters.load import load

import numpy as np

# Create the mnist model.
model = Network(128, ReLU, 128, ReLU, 10, LogSoftMax)

# Assign it some training/testing data.
# Assign it some training/testing data.
model.training.data = dataset
model.labels = labels
model.show = show
model.parameters = load("number_parameters.npy")

#-----------------------#
# Normalize the inputs: #
#-----------------------#

# Keep track of the training mean and variance.
model.anti_momentum = 0.001
model.mean = 0.0
model.variance = 1e-3
model.weight = 0.0

def recenter(x, y):
    x = x - model.mean / model.weight
    return x, y

def rescale(x, y):
    x /= np.sqrt(model.variance / model.weight)
    return x, y

def normalize(x, y):
    return rescale(*recenter(x, y))

print(f'Validation Accuracy: {model.accuracy(*normalize(model.validation.data[0],model.validation.data[1]))}')

