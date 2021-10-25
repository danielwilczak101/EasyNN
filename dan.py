from EasyNN.examples.mnist.fashion.trained import model
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.download import download

download("dress.jpg","https://bit.ly/3b7rsXF")

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=5,
    resize=(28, 28),
    rotate=0,
)

# Converting your image into the correct format for the mnist fashion dataset.
image = image("dress.jpg").format(**format_options)

print(model.classify(image))

model.show(image)

