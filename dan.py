"""from EasyNN.model import Model
from EasyNN.examples.cifar10.untrained import model

model.train()
model.save("cifar10")
model = Model.load("cifar10")"""

from EasyNN.examples.mnist.number.trained import model
from EasyNN.examples.mnist.number.data import dataset

images, labels = dataset


#print(model.classify(images[1]))