import numpy as np
from os.path import exists
from EasyNN.utilities.download import download

def load(file_path: str, url = None):
    """Used for loading model files that have been downloaded.
    
    Args:
        file_path: Path to file to be loaded.
        url (optional): If you also want to download the file
        
    Returns:
        Returns a numpy list of all variables stored in the file.

    Example:
        >>> data = load("model/mnist.npz")
        >>> print(list(data.keys()))
        ['parameters', 'mean', 'variance']
    """
    # If it has already been downloaded skip.
    if not exists("file_path"):
        # If the file doesnt exist but user has given a url.
        if url:
            download(file_path,url)
    # Load the data.
    return np.load(file_path)


def save(file_name :str,**kwargs:dict) -> None:
    """Used for saving model parameters that can been downloaded/used later.
    
    Args:
        file_name: File name is to be saved under.
        variables: Dictinary of numpy arrays to be saved.

    Example:
        >>>
        >>> dct = dict(
        >>>    parameters = model.parameters,
        >>>    mean = model.mean,
        >>>    variance = model.variance
        >>> )
        >>> save("model",dct)
        All variables saved to file named model.npz
    """

    np.savez_compressed(file_name, **kwargs)
    print("Parameters saved.")   