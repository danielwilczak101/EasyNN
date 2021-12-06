from EasyNN.examples.mnist.fashion.structure import model
from EasyNN.examples.mnist.fashion.data import dataset
from EasyNN.examples.mnist.fashion import labels, show
from EasyNN.callbacks import ReachValidationAccuracy
from EasyNN.optimizer import MomentumDescent
import EasyNN.callbacks as cb
import numpy as np

# Assign it some training/testing data.
model.training.data = dataset

# Set when to terminate point. 
# In this case it will end once your validation accuracy hits above 90% three times.
model.callback(ReachValidationAccuracy(limit=0.90, patience=3))

# Used for plotting
model.validation_lr = 0.3
model.validation.accuracy = []

# Establish the labels and show feature.
model.labels = labels
model.show = show

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

model.callback(
    # Set when to terminate point. 
        # In this case it will end once your validation accuracy hits above 90% five times.
    cb.ReachValidationAccuracy(limit=0.30, patience=2),
)

# When the model hit a validation point it will print the iteration and accuracy of the model.
model.print.on_validation_start(iteration=True,accuracy=True)
# When the model completes 10 iterations. It will print that iteration number.
model.print.on_training_start(iteration=True, frequency=10)




