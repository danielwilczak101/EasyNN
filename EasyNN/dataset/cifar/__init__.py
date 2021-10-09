import numpy as np

from EasyNN.utilities.download import download
from EasyNN.utilities.data.load import load as load_data
from EasyNN.utilities.parameters.load import load as load_model

model_filename = "cifar10.model"
data_filename  = "cifar10.npz"
model_url = "#"
data_url  = "https://github.com/danielwilczak101/EasyNN/raw/datasets/cifar/cifar10.npz"
labels = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
}

def __getattr__(name):
    """Used to give the user more understanding names while loading the features."""
    if name == "dataset":
        download(data_filename,data_url)
        return load_data(data_filename)
    if name == "labels":
        return labels
    raise AttributeError(f"module {__name__} has no attribute {name}")