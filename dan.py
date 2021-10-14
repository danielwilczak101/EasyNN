from EasyNN.dataset.mnist.number import dataset
from examples.mnist.number.trained import model

images, labels = dataset

print(model.accuracy(images,labels))