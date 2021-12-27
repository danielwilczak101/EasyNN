from EasyNN.examples.cifar10.trained import model
from EasyNN.examples.cifar10.data import dataset

images, labels = dataset

# Show the image
model.show(images[2])