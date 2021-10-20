from clint.textui import progress
from os.path import exists
import requests

def download(file_name: str,  url: str) -> str:
    """Used for downloading dataset files to be used in the models.

    Args:
        url: Url to file to be downloaded.
        file_name: Name for file to be saved under.

    Returns:
        file_name: Returns the name of the file downloaded.
        File in the same directory where it is being run.

    Example:
        >>> from EasyNN.utilities.download import download
        >>> download("four.jpg","https://bit.ly/3lAJrMe")
        Downloading - four.jpg:
        [################################] 1768/1768 - 00:00:00
        
        File now stored in running file directory.
    """

    if not exists(file_name):
        # Tell the user what they are downloading.
        print(f"Downloading - {file_name}:")
        # Grab the file.
        r = requests.get(url, stream=True)
        # Produces the progress bar while downloading.
        with open(file_name, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
    
    # Give the user the file name if they want to use it in anther function.
    return file_name




