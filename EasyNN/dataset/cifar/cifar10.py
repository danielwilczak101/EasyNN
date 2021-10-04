from os.path import exists
from urllib import request

from EasyNN.dataset.common import *
from EasyNN.model.network import *
from EasyNN.model.layer import *
from EasyNN.model.activation import *
from EasyNN.optimizer import *
from EasyNN.loss import *
from EasyNN.accuracy import *

model_filename = "cifar10.model"
model_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/cifar/cifar10.model"

data_filename = "cifar10.npz"
data_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/cifar/cifar10.npz"

# Label index to label name relation
dataset_labels = {
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

def model(user_image) -> int:
    """Main function that creates the model for the MNIST dataset."""

    # Instantiate the Network
    model = Network()
    # Add layers
    model.add(Layer_Dense(3072, 256))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(256, 256))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(256, 10))
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
        y1, x1, x2, y2 = load(data_filename)
        # Train the model
        model.train(x1, y1, validation_data=(x2, y2), epochs=10000)
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
    prediction = dataset_labels[predictions[0]]

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

        


