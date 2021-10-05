import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from EasyNN.dataset.mnist.number import  dataset, trained_model

from PIL import Image, ImageOps
from urllib import request
from os.path import exists

# Used for common functions that are used across multiple model files.

def preprocess(image_path:str) -> list[int]:
    """Used for taking in a user image and preprocessing 
    it to look similar to the dataset image.

    TODO:
    Args:
        image_path: The file path where the image can be found.
    
    Returns:
        An image that has been preprocessed and is of size [1,784].

    Raises:
        If the file doesnt exist then tell the user you cant find the image.
     """

    # We opened the image and converted to grey scale
    user_img = Image.open("four.jpg").convert('L')
    # We inverted the color to look like our dataset images
    user_img = ImageOps.invert(user_img)
    # Resized the large image so it's shape is like our dataset
    user_img = np.rot90(np.array(user_img.resize([28, 28])), k=3)


    # Filtered out all the grey noise in our image
    user_img = np.where(user_img < 150,0,user_img)

    #refine ^^





    # We reshaped to fit the input of the NN and fed it through the model
    print(trained_model(user_img.reshape((1,784))))

    # Show the array data and actual plotted image.
    show(user_img, "array")  
    show(user_img, "image")

    #Class variables
    opt_size = [28, 28]

    #Check image and saves as variable
    try:
        user_img = Image.open(image_path)
    except IOError:
        pass

    #Checks to see if image is correct size, and if not, shrinks it down to the correct size
    if user_img.size[0] == opt_size[0] & user_img.size[1] == opt_size[1]:
        print('Image is already 28 x 28')
    else:
        processed_img = user_img.resize([28, 28])

    pass

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
    """

    if image_type is None or image_type == "image":
        # Should check if the image is [28,28] or [1,784] 
        plt.imshow(user_image.reshape((28, 28)), cmap='gray')
        plt.show()
    elif image_type == "array":
        np.set_printoptions(linewidth=114)
        print(user_image)
        np.set_printoptions(linewidth=75)


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

