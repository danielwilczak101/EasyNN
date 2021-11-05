from EasyNN.model import Network, ReLU, LogSoftMax

import EasyNN.callbacks as cb
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
    16, ReLU,
    2, LogSoftMax,
)

# Set your models data
model.training.data = (x, y)

# Change to a small learning rate.
model.optimizer.lr = 0.01

model.callback(
    # Set when to terminate point. 
    # In this case it will end once your validation accuracy hits above 90% five times.
    cb.ReachValidationAccuracy(limit=0.90, patience=3),
    # Plot various metrics.
    cb.PlotValidationAccuracy(),
    cb.PlotValidationLoss(),
    cb.PlotTrainingAccuracy()
)

# Print every 20 iterations.
model.on_training_start(
    cb.Printer(
        iteration=True,
        frequency=20)
)
# On each validation step, print the training and validation loss/validation.
model.on_validation_start(
    cb.Printer(
        #training_loss=True,
        #training_accuracy=True,
        #validation_loss=True,
        validation_accuracy=True
        )
)
# At the end during testing, check all of the losses.
model.on_testing_start(
    cb.Printer(
        #training_loss=True,
        #training_accuracy=True,
        #validation_loss=True,
        #validation_accuracy=True,
        testing_loss=True,
        testing_accuracy=True
    )
)

# Always at the end of your setup
model.train()

# Save your model so that you can use it later.
model.save("xor")