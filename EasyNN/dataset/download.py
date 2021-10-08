from urllib import request
from os.path import exists

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
