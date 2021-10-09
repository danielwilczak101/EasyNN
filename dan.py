
from EasyNN.model import Network, ReLU, LogSoftMax
from EasyNN.dataset.mnist.number import dataset, labels
from EasyNN.utilities.image.image import Image
from EasyNN.utilities.image.mnist.show import show
from EasyNN.utilities.parameters.save import save
from EasyNN.utilities.parameters.load import load

import numpy as np

# Create the mnist model.
model = Network(128, ReLU, 128, ReLU, 10, LogSoftMax)

# Assign it some training/testing data.
# Assign it some training/testing data.
model.training.data = dataset
model.labels = labels
model.show = show
model.save = save
#print(np.shape(load("number_parameters.npy")))
model.save = save

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
    model.stop_training |= model.training.iteration >= 500 - 1

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
    print(f'Validation Accuracy: {model.accuracy(*normalize(model.validation.data[0],model.validation.data[1]))}')

# Noramlize other inputs, but don't update the mean and variance.
@model.on_testing_start
def callback():
    model.training.sample = normalize(*model.training.sample)
    model.validation.sample = normalize(*model.validation.sample)
    model.testing.sample = normalize(*model.testing.sample)
    #print(f"  {model.loss(*model.training.sample)     = }")
    #print(f"    {model.loss(*model.validation.sample) = }")
    #print(f"      {model.loss(*model.testing.sample)  = }")
    print(f'Model Accuracy: {model.accuracy(*normalize(model.training.data[0],model.training.data[1]))}')

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

#model.save(,model.parameters)

#image = Image("EasyNN/dataset/mnist/number/images/four.jpg").format(grayscale=True,invert=True,process=True,contrast=30,resize=[28,28],rotate=3)

#print(model.labels[model.classify(image)])

#model.show(image)
