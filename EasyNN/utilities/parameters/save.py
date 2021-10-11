import numpy as np

from EasyNN.typing import Array1D

def save(parameters:Array1D, file_name :str):
    """Used for saving model parameters that can been downloaded/used later.
    
    Args:
        parameters: Numpy array with shape [1,N] filled with model parameters.
        file_name: File name is to be saved under.
        
    Returns:
        A file saved with the paramaters with the name specified in file_name.

    Example:
        >>>
        >>> 
        >>> save(model.parameters, "mnist_paramaters.npy")
        Parameters saved to file named mnist_paramaters.npy
    """
    
    with open(file_name, 'wb') as f:
        np.save(f, parameters)
    print("Parameters saved.")