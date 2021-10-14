from EasyNN.model import Network, Normalize, ReLU, LogSoftMax
from EasyNN.dataset.mnist.number.extras import labels, show
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
arrays = load("model.npz","https://bit.ly/3FQlTLa")

# Set all the models trained parameters.
model.parameters = arrays['parameters']
model.layers[0]._mean = arrays['mean']
model.layers[0]._variance = arrays['variance']
model.layers[0]._weight = 1.0

# Establish the models labels
model.labels = labels

# Use the pre-created show function to 
model.show = show
