from EasyNN.examples.mnist.number.untrained import model
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.download import download
from EasyNN.utilities.parameters import save

# Once training has been finished save the model to use later
@model.on_testing_end
def save_data(model):
    save("model.npz", **model.get_arrays())

# Train the model.
model.train()

# Download and example image.
file = "four.jpg"
url = "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/examples/four.jpg"
download(file, url)

# Establish formating options
format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

# Converting your image into the correct format for the mnist number dataset.
image = image(file).format(**format_options)

# Tell me what the image is.
print(model.classify(image))