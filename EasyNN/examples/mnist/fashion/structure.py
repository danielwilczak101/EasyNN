from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax

# Create the Neural Network Model.
model = Network(
    Normalize(1e-3),
    Randomize(0.5), 800, ReLU,
    Randomize(0.1), 10,  LogSoftMax
)