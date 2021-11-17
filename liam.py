from EasyNN.model import Weight
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.download import download
from EasyNN.examples.mnist.number.trained import model

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

model.show(image)

for layer in model.layers:
    if isinstance(layer, Weight):
        print(layer.parameters)
