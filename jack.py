from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.dataset.cifar import dataset
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

# Create the cifar model.
model = Network(Normalize(3e-3), Randomize(0.2), 256, ReLU, Normalize, Randomize(0.03), 64, ReLU, Normalize, Randomize(0.01), 10, LogSoftMax)

# Assign it some training/testing data.
model.training.data = dataset

# Increase the default learning rate.
model.optimizer.lr *= 2

#===================#
# Create callbacks: #
#===================#

@model.on_training_start
def callback():
    print(f"  {model.training.iteration = }")

@model.on_validation_start
def callback():
    print(f"  {model.loss(*model.training.sample)     = }")
    print(f"    {model.loss(*model.validation.sample) = }")

@model.on_testing_start
def callback():
    print(f"  {model.loss(*model.training.sample)     = }")
    print(f"    {model.loss(*model.validation.sample) = }")
    print(f"      {model.loss(*model.testing.sample)  = }")

model.validation_lr = 0.3
model.validation_losses = []
model.validation_smooth = []
model.validation_average = 0.0
model.validation_weight = 0.0

@model.on_validation_start
def callback():
    model.validation_losses.append(model.loss(*model.validation.sample))
    model.validation_weight -= model.validation_lr * (model.validation_weight - 1)
    model.validation_average -= model.validation_lr * (model.validation_average - model.validation_losses[-1])
    model.validation_smooth.append(model.validation_average / model.validation_weight)

@model.on_testing_start
def callback():
    plt.plot(model.validation_losses)
    plt.plot(model.validation_smooth)
    plt.title("validation / iteration")
    plt.show()

@model.on_validation_start
def callback():
    print("You have 3 seconds to use `[control] + C` to pause the program...")
    print("If you do, use `cont` to continue running or `quit()` to stop.")
    print("Use this to set `model.stop_training = True` to stop training.")
    sleep(3)

print("Type `cont` to start:")
breakpoint()
prompt = "Type `cont` to start:"

# Train the model.
model.train()

print("Any future things you might want to check out at the end of the program?")
print("Type 'cont' to finish the program.")
breakpoint()
prompt = "Type 'cont' to finish the program."
