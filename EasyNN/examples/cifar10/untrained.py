from EasyNN.examples.mnist.fashion.structure import model
from EasyNN.examples.cifar10.data import dataset
from EasyNN.examples.cifar10 import labels, show
from EasyNN.loss.abc import Loss
from EasyNN.optimizer import MomentumDescent
from EasyNN.batch import MiniBatch
import EasyNN.callbacks as cb
import numpy as np


# Set your models data
model.training.data = dataset
model.validation.batch = MiniBatch(256)

# Establish the labels and show feature.
model.labels = labels
model.show = show

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

# Change the default learning rate.
model.optimizer.lr = 0.3

model.callback(
    # Set when to terminate point. 
        # In this case it will end once your validation accuracy hits above 90% five times.
    cb.ReachValidationAccuracy(limit=0.90, patience=2),
)

# When the model hit a validation point it will print the iteration and accuracy of the model.
model.print.on_validation_start(iteration=True, accuracy=True)
# When the model completes 10 iterations. It will print that iteration number.
model.print.on_training_start(iteration=True, frequency=10)
