import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from urllib import request
from PIL import Image, ImageOps

from EasyNN.dataset.common import *
from EasyNN.model.network import *
from EasyNN.model.layer import *
from EasyNN.model.activation import *
from EasyNN.optimizer import *
from EasyNN.loss import *
from EasyNN.accuracy import *

model_filename = "number.model"
model_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/fashion_/fashion.model"

data_filename = "number.npz"
data_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/fashion_/fashion.npz"

# Label index -> Label name
dataset_labels = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
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
        # Get the data
        x1, y1, x2, y2 = load(data_filename)
        # Train the model
        model.train(x1, y1, validation_data=(x2, y2), epochs=100)
        model.evaluate(x2,y2)
        # If you want to save your model
        model.save_parameters(model_filename)

    # load the model to make it better.
    model.load_parameters(model_filename)
    # Predict on the image
    confidences = model.predict(user_image)
    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)
    # Get label name from label index
    return dataset_labels[predictions[0]]



def preprocess(image_path:str, rotate: int = 0, offset: int = 30) -> list[int]:
    """Used for taking in a user image and preprocessing 
    it to look similar to the 28x28  dataset image.

    Args:
        image_path: The file path where the image can be found.
        rotate: If the image needs to be rotated by a multiple of 90 degrees.
            Ex. rotate = 1 -> rotate image 90 degrees
            Ex. rotate = 3 -> rotate image 270 degrees
        offset: How much addition pixel strength you want to remove from the array.
            Ex. original = [10,20,30] if offest = 5 then new_array = [5,15,25]  
    
    Returns:
        An image that has been preprocessed and is of array shape [1,784].

    Raises:
        If the file doesnt exist then tell the user you cant find the image.
     """

    try: 
        # We opened the image and converted to grey scale.
        user_img = Image.open(image_path).convert('L')
        # We inverted the color to look like our dataset images.
        user_img = ImageOps.invert(user_img)
        # Resized the large image so it's shape is like our dataset and rotate using k.
        user_img = np.rot90(np.array(user_img.resize([28, 28])), k=rotate)
        # Filtered out all the grey noise in our image using the mean value.
        user_img = np.where(user_img < np.mean(user_img) + offset,0,user_img)
        # Reshape so its like a dataset image.
        return user_img.reshape((1,784))
    except FileNotFoundError:
        print("File not found.")



def show(user_image:list[int], image_type: str = None) -> None:
    """Show the image as either array data or a matplotlib image.
    
    Args:
        user_image: Users image or image from dataset.
        image_type: Tells whether to show the image as an image or print out.
            image: Show as matplotlib image. (default use.)
            array: Show as numpy formated array print out.

    Returns:
        Either an matplotlib graphed image or numpy print of the image pixel values.

    TODO:
        Rasises:
            Check to make sure the image is the correct size of either [28,28] or [1,784]
                If its not tell them to try using preprocessing()
                to change it to the proper size. 
    """


    if image_type is None or image_type == "image":
        try:
            plt.imshow(user_image.reshape((28, 28)), cmap='gray')
            plt.show()
        except ValueError:
            print(
                "Array not in the correct size of [1,784] or [28,28]. Try, \n \
                    \
                    image = preprocess(file_path)\n \
                    or \n \
                    image = resize(image, (28, 28)) \n "
            )

    elif image_type == "array":
        np.set_printoptions(linewidth=114)
        print(user_image)
        np.set_printoptions(linewidth=75)


def __getattr__(name):
    """Used to give the user more understanding names while loading the features."""
    if name == "model":        
        return model
    if name == "dataset":
        # Download and return the dataset variables.
        download(
            "number.npz",
            "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/number_/number.npz"
        )
        return load("number.npz")

    if name == "trained_model":
        # Download the model and return the model class to be used.
        download(
            "number.model",
            "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/number_/number.model"
        )
        return model
        
    raise AttributeError(f"module {__name__} has no attribute {name}")

