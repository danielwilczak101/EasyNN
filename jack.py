from EasyNN.model import Network, ReLU, LogSoftMax
from EasyNN.dataset.cifar import dataset
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

print("Type `cont` to start:")
breakpoint()

# Create the cifar model.
model = Network(256, ReLU, 64, ReLU, 10, LogSoftMax)

# Assign it some training/testing data.
model.training.data = dataset

# Increase the default learning rate.
model.optimizer.lr *= 3

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
model.anti_momentum = 0.03
model.mean = 0.0
model.variance = 1e-3
model.weight = 0.0

def recenter(x):
    return x - model.mean / model.weight

def rescale(x):
    return x * np.sqrt(model.weight / model.variance)

def normalize(x):
    return rescale(recenter(x))

# Update the mean and variance when training.
@model.on_training_start
def callback():
    # Normalize and update.
    x, y = model.training.sample
    model.weight -= model.anti_momentum * (model.weight - 1)
    # Recenter the training input.
    model.mean -= model.anti_momentum * (model.mean - x.mean(axis=0))
    x = recenter(x)
    # Rescale the training input.
    model.variance -= model.anti_momentum * (model.variance - (x ** 2).mean(axis=0))
    model.training.sample = rescale(x), y

@model.on_validation_start
def callback():
    model.validation.sample = normalize(model.validation.sample[0]), model.validation.sample[1]

@model.on_testing_start
def callback():
    model.testing.sample = normalize(model.testing.sample[0]), model.testing.sample[1]

#----------------------------#
# Apply noise to the inputs: #
#----------------------------#

model.noise = 3e-1

@model.on_training_start
def callback():
    # Apply data augmentation by adding noise to the training samples.
    scale = model.noise / (1 + 1e-2 * model.training.iteration) ** 0.2
    x, y = model.training.sample
    x = x + np.random.normal(scale=scale, size=x.shape)
    model.training.sample = (x, y)

#------------------#
# Collect results: #
#------------------#

@model.on_validation_start
def callback():
    print(f"  {model.loss(*model.training.sample)     = }")
    print(f"    {model.loss(*model.validation.sample) = }")

# Uncomment the following line to enable editing as the model trains:
#@model.on_validation_start
def callback():
    print("You have 10 seconds to use `[control] + C` to pause the program...")
    print("If you do, use `cont` to continue running or `quit()` to stop.")
    sleep(10)

@model.on_testing_start
def callback():
    print(f"  {model.loss(*model.training.sample)     = }")
    print(f"    {model.loss(*model.validation.sample) = }")
    print(f"      {model.loss(*model.testing.sample)  = }")

model.validation_losses = []
model.validation_smooth = []
model.validation_average = 0.0
model.validation_weight = 0.0

@model.on_validation_start
def callback():
    model.validation_losses.append(model.loss(*model.validation.sample))
    model.validation_weight -= model.anti_momentum * (model.validation_weight - 1)
    model.validation_average -= model.anti_momentum * (model.validation_average - model.validation_losses[-1])
    model.validation_smooth.append(model.validation_average / model.validation_weight)

@model.on_testing_start
def callback():
    plt.plot(model.validation_losses)
    plt.plot(model.validation_smooth)
    plt.title("validation / iteration")
    plt.show()

# Train the model.
model.train()
