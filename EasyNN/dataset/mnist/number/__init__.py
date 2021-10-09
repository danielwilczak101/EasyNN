import numpy as np

from EasyNN.utilities.download import download
from EasyNN.utilities.data.load import load as load_data
from EasyNN.utilities.parameters.load import load as load_model

model_filename = "number.model"
data_filename  = "number.npz"
model_url = "#"
data_url  = "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/number.npz"

labels = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
}

def __getattr__(name):
    """Used to give the user more understanding names while loading the features."""
    if name == "dataset":
        download(data_filename,data_url)
        return load_data(data_filename)
    if name == "labels":
        return labels
    raise AttributeError(f"module {__name__} has no attribute {name}")