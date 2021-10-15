from examples.mnist.number.trained import model
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.image.compare import compare
from EasyNN.utilities.download import download
from EasyNN.dataset.mnist.number.data import dataset


# Download an example image.
download("four.jpg","https://bit.ly/3lAJrMe")

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

# Converting your image into the correct format for the mnist number dataset.
image = image("four.jpg").format(**format_options)

compare(image,image)

print(model.classify(image))
