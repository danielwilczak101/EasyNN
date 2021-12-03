from EasyNN.model import Model
from EasyNN.utilities import download

# Required for the trained model.
download("number_parameters.npz", "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/number_parameters.npz")
download("number_structure.pkl", "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/number_structure.pkl")

# Load the saved model
model = Model.load("number")