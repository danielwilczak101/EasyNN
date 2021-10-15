from examples.mnist_numbers_untrained_model import model
from EasyNN.utilities.image.image import Image
from EasyNN.utilities.download import download

file = "four.jpg"
url = "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/examples/four.jpg"

# Download an example image.
download(file, url)

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

# Converting your image into the correct format for the mnist number dataset.
image = Image(file).format(**format_options)

print(model.classify(image))
