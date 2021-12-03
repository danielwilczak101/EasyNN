from EasyNN.examples.cifar10.untrained import model
from EasyNN.utilities import Preprocess, download

try:
    model.train()
    model.save("cifar10")
except:
    model.save("cifar10")


# Download an example image.
download("ship.png","https://bit.ly/31jyerQ")

format_options = dict(
    contrast=0,
    resize=(32, 32),
    rotate=1,
    flatten='F'
)

# Converting your image into the correct format for the mnist number dataset.
image = Preprocess("ship.png").format(**format_options)

# Show the image after it has been processed.
model.show(image)

# Classify what the image is using the pretrained model.
print(model.classify(image))