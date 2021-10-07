import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from urllib import request
from os.path import exists

# Used for common functions that are used across multiple model files.

def make_model(model, model_filename, data_filename, data_url,epochs):
    """Used for training a skelition NN."""
    
    # Download the dataset and setup for model.
    if not exists(data_filename):
        download(data_filename, data_url)

    model.dataset = load(data_filename)
    x1, y1, x2, y2 = model.dataset
    # Train the model
    model.train(x1, y1, validation_data=(x2, y2), epochs=epochs)
    model.evaluate(x2,y2)
    # If you want to save your model
    model.save_parameters(model_filename)
    model.load_parameters(model_filename)

    return model


def make_trained_model(model, model_filename, model_url,data_filename,data_url):
    """Used for creating the trained model."""

    if not exists(model_filename):
        download(model_filename, model_url)
    if not exists(data_filename):
        download(data_filename, data_url)
    
    model.dataset = load(data_filename)
    model.load_parameters(model_filename)

    return model

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

