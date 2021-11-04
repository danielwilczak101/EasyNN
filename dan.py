from EasyNN.model import Network, ReLU, LogSoftMax
from EasyNN.callbacks import Printer, PlotValidationAccuracy, PlotValidationLoss, ReachValidationAccuracy
import numpy as np

# Setup the XOR dataset.
x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
] * 100, dtype=int)

y = np.array([0,1,1,0] * 100, dtype=int)

# Create the Neural Network Model.
model = Network(
    4, ReLU,
    2, LogSoftMax,
)

# Set your models data
model.training.data = (x, y)

model.callback(PlotValidationAccuracy())
model.callback(PlotValidationLoss())

# Set when to terminate point. 
# In this case it will end once your validation accuracy hits above 90% five times.
model.callback(ReachValidationAccuracy(limit=0.90, patience=3))

# Set the learning rate
model.optimizer.lr = 0.01


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


# Always at the end
model.train()