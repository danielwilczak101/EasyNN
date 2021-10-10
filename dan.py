
from EasyNN.model import Network, ReLU, LogSoftMax
from EasyNN.dataset.mnist.number import dataset, labels
from EasyNN.utilities.image.image import Image
from EasyNN.utilities.image.mnist.show import show
from EasyNN.utilities.parameters.load import load

import numpy as np

# Create the mnist model.
model = Network(128, ReLU, 128, ReLU, 10, LogSoftMax)

# Assign it some training/testing data.
model.training.data = dataset
model.labels = labels
model.show = show


model(model.training[0][0])
model.parameters = load("mnist_paramaters.npy")
model.validation.data = dataset
print(f'Validation Accuracy: {model.accuracy(model.validation.data[0],model.validation.data[1])}')

image = Image("EasyNN/dataset/mnist/number/images/four.jpg").format(grayscale=True,invert=True,process=True,contrast=30,resize=[28,28],rotate=3)

print(model.classify(image))

model.show(image)