from EasyNN.examples.mnist.fashion.structure import model
from EasyNN.examples.mnist.number.data import dataset
from EasyNN.examples.mnist.number import labels, show
from EasyNN.optimizer import MomentumDescent
from EasyNN.batch import MiniBatch

import EasyNN.callbacks as cb
import numpy as np

# Set your models data
model.training.data = dataset

# Establish the labels and show feature.
model.labels = labels
model.show = show 

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

# Change the default learning rate.
model.optimizer.lr = 0.03

# Test against 1024 validation images to see accuracy.
def setup(model):
    model.validation.batch = MiniBatch(1024)

model.callback(
    # Set when to terminate point. 
        # In this case it will end once your validation accuracy hits above 90% five times.
    cb.ReachValidationAccuracy(limit=0.95, patience=2),
)

# When the model hit a validation point it will print the iteration and accuracy of the model.
model.print.on_validation_start(iteration=True,accuracy=True)
# When the model completes 10 iterations. It will print that iteration number.
model.print.on_training_start(iteration=True, frequency=10)

