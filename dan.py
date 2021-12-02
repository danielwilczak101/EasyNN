"""
from EasyNN.model import Model
from EasyNN.examples.cifar10.untrained import model

model.train()
model.save("cifar10")
model = Model.load("cifar10")

"""

from EasyNN.examples.mnist.fashion.trained import model
from EasyNN.examples.mnist.fashion.data import dataset

images, labels = dataset

# Classify what the second image is in the dataset.
print(model.classify(images[0]))

model.show(images[0])

model.accuracy(dataset)