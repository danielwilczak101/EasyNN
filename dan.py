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
        # Add smooth=False for plots.
    cb.PlotValidationAccuracy(),
    cb.PlotValidationLoss(),
    cb.PlotTrainingAccuracy()
)

# Current form of the printer.
model.on_training_start(
    cb.Printer(start="\n", iteration=True, end="", frequency=10)
)

model.on_validation_end(
    cb.Printer(indent=0, start=", ", validation_accuracy=True, end="")
)



# Always at the end of your setup
model.train()

# Save your model so that you can use it later.
model.save("xor")