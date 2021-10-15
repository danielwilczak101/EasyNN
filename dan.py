from examples.mnist.number.trained import model
from EasyNN.utilities.image.image import Image
from EasyNN.utilities.download import download

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
image = Image("four.jpg").format(**format_options)

print(model.classify(image))
print()