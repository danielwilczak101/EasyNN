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


# model.plot(validation_accuracy=True, validation_loss=True, training_accuracy=True, axis="epoch")
model.print.on_validation_start(iteration=True,accuracy=True)
model.print.on_training_start(iteration=True, frequency=10)

# Always at the end of your setup
#with model.save_with("xor"), model.plot_with(...):
#    model.train()

try:
    model.train()
    model.save("xor")
except KeyboardInterrupt:
    model.save("xor")

# Save your model so that you can use it later.
#model.save("xor")

#model.plot()