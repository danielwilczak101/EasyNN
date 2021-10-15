from EasyNN.examples.mnist.number.trained import model
from EasyNN.examples.mnist.number import show
from EasyNN.examples.mnist.number.data import dataset
from EasyNN.utilities.download import download
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.image.compare import compare
from EasyNN.utilities.download import download

images, labels = dataset

download("six.png","https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/examples/six.png")

print(model.accuracy(images, labels))

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=35,
    resize=(28, 28),
    rotate=0,
)

# Converting your image into the correct format for the mnist number dataset.
image = image("six.png").format(**format_options)

show(image)

print(model.classify(image))




