from EasyNN.utilities.download import download
from EasyNN.utilities.data.load import load 
from os.path import exists 

file = "numbers.npz"
url =  "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/number.npz"

def __getattr__(name):
    """Used to give the user more understanding names while loading the features."""
    if name == "dataset":
        if not exists(file):
            download(file,url)
        return load(file)
    raise AttributeError(f"module {__name__} has no attribute {name}")
