from EasyNN.examples.cifar10.trained import model
from EasyNN.examples.cifar10.data import dataset


def hello():
  images, labels = dataset

  # Show the image
  model.show(images[2])
