from EasyNN.examples.cifar10.untrained import model
from EasyNN.examples.cifar10.data import dataset

images, labels = dataset

try:
    model.train()
except KeyboardInterrupt:
    model.save("cifar10")