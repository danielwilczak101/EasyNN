from examples.mnist.number.trained import model
from EasyNN.dataset.mnist.number.data import dataset

images,labels = dataset

print(model.accuracy(images,labels))