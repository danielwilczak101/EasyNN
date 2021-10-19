from EasyNN.utilities.parameters import load
from EasyNN.utilities.download import download
from EasyNN.examples.mnist.number.data import dataset
import numpy as np
import pickle

x,y = dataset

file = open(download("number_model.pkl", "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/number_model.pkl"), 'rb')     
model = pickle.load(file)
file.close()

model(np.empty(28 * 28))
model.set_arrays(**load("number_model.npz", "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/number_model.npz"))

print(model.accuracy(x,y))