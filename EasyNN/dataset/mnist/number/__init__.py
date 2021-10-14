from EasyNN.utilities.download import download
from EasyNN.utilities.data.load import load
from .extras import file, url
from os.path import exists

if not exists(file):
    download(file, url)
dataset = load(file)
