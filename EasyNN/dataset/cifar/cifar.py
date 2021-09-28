import gzip
import pickle
from os.path import exists
from urllib import request
import matplotlib.pyplot as plt
import numpy as np

from EasyNN.dataset.common import *
from EasyNN.model.network import *
from EasyNN.model.layer import *
from EasyNN.model.activation import *
from EasyNN.optimizer import *
from EasyNN.loss import *
from EasyNN.accuracy import *

dataset_files = [
        ["data","cifar-10-python.tar"]
]

model_file = ['Trained CIFAR model','cifar.model']

def model(user_image) -> int:
    """Main function that creates the model for the MNIST dataset."""

    # Instantiate the Network
    model = Network()
    # Add layers
    model.add(Layer_Dense(1024, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Set loss, optimizer and accuracy objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-3),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    model.finalize()

    # Check model file exists:
    if exists(model_file[1]) == False:
        # Get the data
        x1, y1, x2, y2 = load()
        # Train the model
        model.train(x1, y1, validation_data=(x2, y2), epochs=100)
        model.evaluate(x1,y1)
        # If you want to save your model
        model.save_parameters(model_file[1])

    # load the model to make it better.
    model.load_parameters(model_file[1])

    # Label index to label name relation
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

    # Predict on the image
    confidences = model.predict(user_image)
    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)
    # Get label name from label index
    return cifar_labels[predictions[0]]


def download_mnist() -> None:
    """Downloads four of the mnist dataset files used for traininig and testing."""
    base_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/number_/data/"
    for name in dataset_files:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def download_trained_model() ->None:
    """Used to check if the pretained model is downloaded if 
    not download the pretrained model from github."""

    if not exists(model_file[1]):
        base_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/number_/trained_models/"  
        print("Downloading "+model_file[0]+"...")
        request.urlretrieve(base_url+model_file[1], model_file[1])
        print("Download complete.")

def save_mnist() -> None:
    """Restrucures the dataset into a more useable dictonary. Saves the
    data to a pickle file to be loaded later."""

    mnist = {}
    for name in dataset_files[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in dataset_files[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("number.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def mnist_download() -> None:
    """Download data set if it doesnt already exist."""
    if any(not exists(file[1]) for file in (dataset_files)):
        download_mnist()
        save_mnist()

def load() -> list[list[int]]:
    """Loads the unpacked pickel data that was saved from converting
     the downloaded byte data."""

    with open("number.pkl",'rb') as f:
        mnist = pickle.load(f)

    return  mnist["data"]
            

def __getattr__(name):
    """Used to give the user more understanding names while loading the features."""
    if name == "model":        
        return model
    if name == "dataset":
        mnist_download()
        return load()
    if name == "trained_model":
        download_trained_model()
        return model
        
    raise AttributeError(f"module {__name__} has no attribute {name}")


        

