
from EasyNN.model import Network, ReLU, LogSoftMax
from EasyNN.dataset.mnist.number import dataset, labels
import numpy as np

# Create the mnist model.
model = Network(128, ReLU, 128, ReLU, 10, LogSoftMax)

# Assign it some training/testing data.
model.training.data = dataset
model.labels = labels

# Reduce the default learning rate.
model.optimizer.lr /= 3

#===================#
# Create callbacks: #
#===================#

@model.on_training_start
def callback():
    print(f"  {model.training.iteration = }")

@model.on_training_end
def callback():
    # Terminate after 10000 iterations.
    model.stop_training |= model.training.iteration >= 10000 - 1

#=============================#
# Data augmentation examples: #
#=============================#

#-----------------------#
# Normalize the inputs: #
#-----------------------#

# Keep track of the training mean and variance.
model.anti_momentum = 0.001
model.mean = 0.0
model.variance = 1e-3
model.weight = 0.0

def recenter(x, y):
    x = x - model.mean / model.weight
    return x, y

def rescale(x, y):
    x /= np.sqrt(model.variance / model.weight)
    return x, y

def normalize(x, y):
    return rescale(*recenter(x, y))

# Update the mean and variance when training.
@model.on_training_start
def callback():
    # Normalize and update.
    x, y = model.training.sample
    model.weight -= model.anti_momentum * (model.weight - 1)
    # Recenter the training input.
    model.mean -= model.anti_momentum * (model.mean - x.mean(axis=0))
    x, y = recenter(x, y)
    # Rescale the training input.
    model.variance -= model.anti_momentum * (model.variance - (x ** 2).mean(axis=0))
    model.training.sample = rescale(x, y)

# Noramlize other inputs, but don't update the mean and variance.
@model.on_validation_start
def callback():
    model.training.sample = normalize(*model.training.sample)
    model.validation.sample = normalize(*model.validation.sample)
    print(f"  {model.loss(*model.training.sample)     = }")
    print(f"    {model.loss(*model.validation.sample) = }")

# Noramlize other inputs, but don't update the mean and variance.
@model.on_testing_start
def callback():
    model.training.sample = normalize(*model.training.sample)
    model.validation.sample = normalize(*model.validation.sample)
    model.testing.sample = normalize(*model.testing.sample)
    print(f"  {model.loss(*model.training.sample)     = }")
    print(f"    {model.loss(*model.validation.sample) = }")
    print(f"      {model.loss(*model.testing.sample)  = }")

#----------------------------#
# Apply noise to the inputs: #
#----------------------------#

@model.on_training_start
def callback():
    # Apply data augmentation by adding noise to the training samples.
    scale = 0.001 / (1 + 1e-2 * model.training.iteration) ** 0.3
    x, y = model.training.sample
    x = x + np.random.normal(scale=scale, size=x.shape)
    model.training.sample = (x, y)

# Train the model.
model.train()

image = dataset[0][0]
label = dataset[1][0]

print(model.labels[model.classify(image)])
print(model.labels[label])

print(f'Accuracy: {model.accuracy(dataset[0][:100],dataset[1][:100])}')