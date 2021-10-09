import numpy as np

def load(file_path: str):
    """Used for loading dataset files that have been downloaded.
    
    Args:
        file_path: Path to file to be loaded.
        
    Returns:
        x: Data used to train models.
        y: Dataset labels.

   Example:
        >>> data,labels = load("model/mnist.npz")
        >>> # Print first dataset example and first label
        >>> print(data[0])
        >>> print(label[0])
        [0 200 ... 15  0]
        5
    """
    
    try:
        with np.load(file_path) as data:
            return  data['x'], \
                    data['y']
    except FileNotFoundError:
        print("File doesnt exist or you haven't downloaded the dataset yet.")