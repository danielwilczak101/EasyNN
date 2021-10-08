import numpy as np

from EasyNN.dataset.download import download
from EasyNN.dataset.load import load

model_filename = "fashion.model"
data_filename = "fashion.npz"
model_url = "#"
data_url = "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/fashion/fashion.npz"
labels = {
    0 :	'T-shirt/top',
    1 :	'Trouser',
    2 :	'Pullover',
    3 :	'Dress',
    4 :	'Coat',
    5 :	'Sandal',
    6 :	'Shirt',
    7 :	'Sneaker',
    8 :	'Bag',
    9 :	'Ankle boot',
}

def __getattr__(name):
    """Used to give the user more understanding names while loading the features."""
    if name == "dataset":
        download(data_filename,data_url)
        return load(data_filename)
    if name == "labels":
        return labels
    raise AttributeError(f"module {__name__} has no attribute {name}")