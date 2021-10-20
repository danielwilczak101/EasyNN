from EasyNN.examples.mnist.number.trained import model
from EasyNN.examples.mnist.number.data import dataset

images, labels = dataset

# Check the model again so its teh same.
print(model.accuracy(images, labels))
print(model.classify(images[0]))