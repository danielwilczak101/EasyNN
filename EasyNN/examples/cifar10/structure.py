from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax

# Create the mnist model.
model = Network(
    Normalize(1e-6), 
    #Randomize(0.01),
    3072, ReLU,
    256, ReLU,
    256, ReLU,
    10, LogSoftMax,
)

