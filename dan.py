from EasyNN.dataset.mnist.number import dataset
from examples.mnist.number.trained import model

data, labels = dataset

print(model.accuracy(data,labels))