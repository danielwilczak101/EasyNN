import gzip
import pickle
from os.path import exists
from urllib import request

from EasyNN.dataset.common import *
from EasyNN.model.network import *
from EasyNN.model.layer import *
from EasyNN.model.activation import *
from EasyNN.optimizer import *
from EasyNN.loss import *
from EasyNN.accuracy import *

model_filename = "fashion.model"
model_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/fashion_/fashion.model"

data_filename = "fashion.npz"
data_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/fashion_/fashion.npz"

# Label index -> Label name
fashion_mnist_labels = {
    0 :	'T-shirt/top',
    1 :	'Trouser',
    2 :	'Pullover',
    3 :	'Dress',
    4 :	'Coat',
    5 :	'Sandal',
    6 :	'Shirt',
    7 :	'Sneaker',
    8 :	'Bag',
    9 :	'Ankle boot',
}

def model(user_image) -> int:
    """Main function that creates the model for the MNIST dataset."""

    # Instantiate the Network
    model = Network()
    # Add layers
    model.add(Layer_Dense(784, 128))
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
    if exists(model_filename) == False:
        # Get the dataset
        x1, y1, x2, y2 = load(data_filename)
        # Train the model
        model.train(x1, y1, validation_data=(x2, y2), epochs=150)
        model.evaluate(x1,y1)
        # If you want to save your model
        model.save_parameters(model_filename)

    # load the model to make it better.
    model.load_parameters(model_filename)
    # Predict on the image
    confidences = model.predict(user_image)
    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)
    # Get label name from label index
    prediction = fashion_mnist_labels[predictions[0]]

    return prediction


def __getattr__(name):
    """Used to give the user more understanding names while loading the features."""
    if name == "model":        
        return model
    if name == "dataset":
        # Download and return the dataset variables.
        download(
            data_filename,
            data_url
        )
        return load(data_filename)

    if name == "trained_model":
        # Download the model and return the model class to be used.
        download(
            model_filename,
            model_url
        )
        return model
        
    raise AttributeError(f"module {__name__} has no attribute {name}")

        


