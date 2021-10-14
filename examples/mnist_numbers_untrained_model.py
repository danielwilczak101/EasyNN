from EasyNN.utilities.image.image import Image
from EasyNN.utilities.download import download
from EasyNN.utilities.parameters import save
from examples.mnist.number.untrained import model

# Once training has been finished save the model to use later
@model.on_testing_end
def save_data(model):
    arrays = dict(
        parameters = model.parameters,
        mean = model.layers[0]._mean,
        variance = model.layers[0]._variance
    )
    save("model.npz",**arrays) 

# Train the model.
model.train()

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
