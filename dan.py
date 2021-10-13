from EasyNN.dataset.mnist.number import trained_model as model
from EasyNN.utilities.image.image import Image

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

file_name = "EasyNN/dataset/mnist/number/images/four.jpg"

# Converting your image into the correct format for the mnist number dataset.
image = Image(file_name).format(**format_options)

model.show(image)

print(model.classify(image))
print(model.accuracy())


