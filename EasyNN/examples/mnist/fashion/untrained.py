from EasyNN.examples.mnist.fashion.data import dataset
from EasyNN.examples.mnist.fashion import labels, show
from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.optimizer import MomentumDescent
from EasyNN.utilities.callbacks.printer import printer
import EasyNN.utilities.callbacks.plot as plot
from EasyNN.typing import Callback
from EasyNN.batch import MiniBatch
import numpy as np


# Create the mnist model.
model = Network(
    Normalize(1e-3),
    Randomize(0.5), 800, ReLU,
    Randomize(0.1), 10,  LogSoftMax
)

# Aim for 90% validation accuracy for 5 validation iterations in a row.
model.validation.accuracy_patience = 1
model.validation.accuracy_limit = 0.85
model.validation.successes = 0
model.validation_lr = 0.3
model.validation.accuracy = []
# Assign it some training/testing data.
model.training.data = dataset

# Extra features.
model.labels = labels
model.show = show

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

@model.on_optimization_start
def setup(model):
    model.validation.batch = MiniBatch(1024)

@model.on_validation_end
def terminate(model):
    if model.accuracy(*model.validation.sample) > model.validation.accuracy_limit:
        model.validation.successes += 1
    else:
        model.validation.successes = 0
    model.stop_training |= model.validation.successes >= model.validation.accuracy_patience

# Print every 20 iterations.
model.on_training_start(printer(iteration=True, frequency=20))
# On each validation step, print the training and validation loss/validation.
model.on_validation_start(printer(training_loss=True, validation_loss=True))
model.on_validation_start(printer(training_accuracy=True, validation_accuracy=True))
# At the end during testing, check all of the losses.
model.on_testing_start(printer(training_loss=True, validation_loss=True, testing_loss=True))
model.on_testing_start(printer(training_accuracy=True, validation_accuracy=True, testing_accuracy=True))

@model.on_validation_start
def save_validation_accuracy(model):
    accuracy = model.accuracy(*model.validation.sample)
    model.validation.accuracy.append(accuracy)

model.on_testing_start(plot.validation.accuracy)


