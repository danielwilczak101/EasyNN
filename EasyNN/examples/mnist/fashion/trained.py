from EasyNN.model import Model
from EasyNN.utilities.download import download

# Required for the trained model.
download("fashion_parameters.npz","https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/fashion/fashion_parameters.npz")
download("fashion_structure.pkl","https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/fashion/fashion_structure.pkl")

# Load the saved model
model = Model.load("fashion")
