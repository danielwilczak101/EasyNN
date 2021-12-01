from EasyNN.model import Model
from EasyNN.utilities.download import download

# Required for the trained model.
download("cifar10_parameters.npz","https://github.com/danielwilczak101/EasyNN/raw/datasets/cifar/cifar10_parameters.npz")
download("cifar10_structure.pkl","https://github.com/danielwilczak101/EasyNN/raw/datasets/cifar/cifar10_structure.pkl")

# Load the saved model
model = Model.load("cifar10")



