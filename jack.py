from EasyNN.examples.mnist.number.untrained import model
from EasyNN.examples.mnist.number.data import dataset
from EasyNN.model import Model

images, labels = dataset

# Train the model.
model.train()
model.save("number")

# Check the models accuracy
print(model.accuracy(images, labels))

# Load the saved model
model = Model.load("number")

# Check the model again so its teh same.
print(model.accuracy(images, labels))
