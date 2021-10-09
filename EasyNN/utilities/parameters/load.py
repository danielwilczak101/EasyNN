import numpy as np

def load(file_path: str):
    """Used for loading model files that have been downloaded.
    
    Args:
        file_path: Path to file to be loaded.
        
    Returns:
        parameters: Returns a numpy array of a pretrained model parameters
        that will later be loaded into a model.

    Example:
        >>> model_parameters = load("model/mnist.npz")
        >>> print(model_parameters)
        [-0.03679278 0.07100084 ... -0.00449589  -0.00858782]
    """

    try:
        np.load(file_path)
    except FileNotFoundError:
        print("File doesnt exist or you haven't downloaded the model yet.")