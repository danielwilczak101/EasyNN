from EasyNN.model import Model
from EasyNN.utilities.download import download

# Required for the trained model.
download("fashion_parameters.npz","#")
download("fashion_structure.pkl","#")

# Load the saved model
model = Model.load("fashion")
