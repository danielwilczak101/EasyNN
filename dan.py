from EasyNN.examples.cifar10.untrained import model
from EasyNN.examples.cifar10.data import dataset

try:
    model.train()
except KeyboardInterrupt:
    model.save("cifar10")