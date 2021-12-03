from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax

# Create the mnist model.
model = Network(
    Normalize(1e-3), Randomize(0.01),
    1024, ReLU,
    256, ReLU,
    10, LogSoftMax,
)