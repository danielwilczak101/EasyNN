import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from urllib import request
from os.path import exists

# Used for common functions that are used across multiple model files.

def download(file_name: str,url: str) -> None:
    """Used for downloading dataset files to be used in the models.

    Args:
        url: Url to file to be downloaded.
        file_name: Name for file to be saved under.

    Returns:
        File in the same directory where it is being run.
    """

    if not exists(file_name):
        print(f"Downloading {file_name}...")
        request.urlretrieve(url, file_name)
        print("Download complete.")

def load(file_path: str):
    """Used for loading dataset files that have been downloaded.
    
    Args:
        file_path: Path to file to be loaded.
        
    Returns:
        train_data: Training data used to train models.
        train_model: Training data labels.
        test_data: Testing data to find out how accurate the model is.
        test_labels: Testing data labels.
    
    Raises:
        FileNotFoundError:
            Suggest them to run the download first."""
    
    try:
        with np.load(file_path) as data:
            return  data['train_data'], \
                    data['train_labels'],\
                    data['test_data'],\
                    data['test_labels']
    except FileNotFoundError:
        print("File doesnt exist or you haven't downloaded the dataset yet.")

