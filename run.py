import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from zipfile import ZipFile

# Load the data into memory
def load_files(filenames):
    for name in filenames:
        with open(name, 'rb') as f:
            mydict = pickle.load(f, encoding='latin1')
    return mydict 


def cifar10_plot(data,labels):

    cifar_labels = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    im_r = data[0:1024].reshape(32, 32)
    im_g = data[1024:2048].reshape(32, 32)
    im_b = data[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    print("shape: ", img.shape)    

    plt.title(cifar_labels[labels])
    plt.imshow(img) 
    plt.show()

""" files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5" 
]

# Grab the files
data = load_files(["test_batch"])
test_data =  data['data']
test_label = data['labels']

train_data = []
train_labels = []

for file in files:
    dataset = load_files(files)
    train_data.extend(dataset['labels'])
    train_labels.extend(dataset['data'])

print(np.shape(train_data))

np.savez_compressed(
    'cifar10',
    train_data=train_data,
    train_labels = train_labels,
    test_data=test_data,
    test_label=test_label
) """
def load():
    with np.load('cifar10.npz') as data:
        return data['train_data'], data['train_labels'],data['test_data'], data['test_label']

train_data,train_labels,test_data,test_label = load()

print(np.shape(train_data))


