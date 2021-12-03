from EasyNN.examples.mnist.number.untrained import model
from EasyNN.examples.mnist.number.data import dataset
from EasyNN.callbacks import ReachValidationAccuracy

images, labels = dataset

# Train the model.
model.callback(ReachValidationAccuracy(limit=0.80, patience=3))
model.train()
model.save("number")

# Check the models accuracy
print(model.accuracy(images, labels))