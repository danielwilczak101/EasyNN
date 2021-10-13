from EasyNN.dataset.mnist.number import model
from EasyNN.utilities.image.image import Image

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

# This file should be downloaded for the user as the example.
file_name = "EasyNN/dataset/mnist/number/images/four.jpg"

# Converting your image into the correct format for the mnist number dataset.
image = Image(file_name).format(**format_options)

dataset_image = model.training.data[0][2]

model.train()
