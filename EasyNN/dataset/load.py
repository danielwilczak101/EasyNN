import numpy as np

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