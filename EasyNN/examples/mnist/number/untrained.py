from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.callbacks import Printer, ReachValidationAccuracy
from EasyNN.examples.mnist.number.data import dataset
from EasyNN.examples.mnist.number import labels, show
from EasyNN.optimizer import MomentumDescent
from EasyNN.typing import Callback
from EasyNN.batch import MiniBatch

import EasyNN.utilities.callbacks.plot as plot
import numpy as np

# Create the mnist model.
model = Network(
    Normalize(1e-3), Randomize(0.01),
    1024, ReLU,
    256, ReLU,
    10, LogSoftMax,
)

# Set your models data
model.training.data = dataset

# Set when to terminate point. 
# In this case it will end once your validation accuracy hits above 90% five times.
model.callback(ReachValidationAccuracy(limit=0.90, patience=3))

# Used for plotting
model.validation_lr = 0.3
model.validation.accuracy = []

# Establish the labels and show feature.
model.labels = labels
model.show = show

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

# Change the default learning rate.
model.optimizer.lr = 0.03

# Test against 1024 validation images to see accuracy.
@model.on_optimization_start
def setup(model):
    model.validation.batch = MiniBatch(1024)

# Print every 20 iterations.
model.on_training_start(Printer(iteration=True, frequency=20))
# On each validation step, print the training and validation loss/validation.
model.on_validation_start(
    Printer(training_loss=True, validation_loss=True),
    Printer(training_accuracy=True, validation_accuracy=True)
)
# At the end during testing, check all of the losses.
model.on_testing_start(
    Printer(training_loss=True, validation_loss=True, testing_loss=True),
    Printer(training_accuracy=True, validation_accuracy=True, testing_accuracy=True)
)

@model.on_validation_start
def save_validation_accuracy(model):
    accuracy = model.accuracy(*model.validation.sample)
    model.validation.accuracy.append(accuracy)

model.on_testing_start(plot.validation.accuracy)
