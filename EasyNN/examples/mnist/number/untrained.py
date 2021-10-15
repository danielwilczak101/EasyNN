from EasyNN.batch import MiniBatch
from EasyNN.examples.mnist.number.data import dataset
from EasyNN.examples.mnist.number import labels, show
from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.optimizer import MomentumDescent
from EasyNN.utilities.momentum import Momentum
import matplotlib.pyplot as plt
import numpy as np

# Create the mnist model.
model = Network(
    Normalize(1e-3),
    Randomize(0.5), 256, ReLU,
    Randomize(0.3), 128, ReLU,
    Randomize(0.1), 10,  LogSoftMax
)

# Assign it some training/testing data.
model.training.data = dataset

# Extra features.
model.labels = labels
model.show = show

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

# Aim for 90% validation accuracy for 5 validation iterations in a row.
model.validation.accuracy_patience = 1
model.validation.accuracy_limit = 0.95
model.validation.successes = 0
