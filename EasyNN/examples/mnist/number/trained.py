from EasyNN.model import Network, Normalize, ReLU, LogSoftMax
from EasyNN.examples.mnist.number import labels, show
from EasyNN.utilities.parameters import load
import matplotlib.pyplot as plt
import numpy as np

# Create the mnist model.
model = Network(
    Normalize,
    256, ReLU,
    128, ReLU,
    10, LogSoftMax
)
# Finilize the model.
model(np.empty(28 * 28))

# Download model.npz from EasyNN dataset branch and load it.
model.set_arrays(**load("model.npz", "https://bit.ly/3FQlTLa"))

# Establish the models labels
model.labels = labels

# Use the pre-created show function to 
model.show = show
