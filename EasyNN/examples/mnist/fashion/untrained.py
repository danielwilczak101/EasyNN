from EasyNN.examples.mnist.fashion import labels, show
from EasyNN.examples.mnist.fashion.data import dataset
from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.optimizer import MomentumDescent

# Create the mnist model.
model = Network(
    Normalize(3e-4),
    Randomize(0.5), 512, ReLU,
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
model.validation.accuracy_patience = 2
model.validation.accuracy_limit = 0.92
model.validation.successes = 0

