from EasyNN.examples.mnist.fashion.untrained import model
from EasyNN.examples.mnist.fashion.data import dataset
from EasyNN.model import Model

images, labels = dataset

# Train the model.
model.train()
model.save("fashion")

# Check the models accuracy
print(model.accuracy(images, labels))

# Load the saved model
model = Model.load("fashion")

# Check the model again so its teh same.
print(model.accuracy(images, labels))
