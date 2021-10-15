from EasyNN.examples.mnist.fashion.trained import model
from EasyNN.examples.mnist.fashion.data import dataset
from EasyNN.examples.mnist.fashion import show

from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.download import download

images, labels = dataset

print(model.accuracy())




