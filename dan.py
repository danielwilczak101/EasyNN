"""
from EasyNN.model import Model
from EasyNN.examples.cifar10.untrained import model

model.train()
model.save("cifar10")
model = Model.load("cifar10")

"""

from EasyNN.examples.cifar10.trained import model
from EasyNN.examples.cifar10.data import dataset

images, labels = dataset

print(model.classify(images[1]))
#"""
