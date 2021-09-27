import matplotlib.pyplot as plt
import numpy as np

# Used for common functions that are used across multiple model files.

def preprocess(image_path:str) -> list[int]:
    """Used for taking in a user image and preprocessing 
    it to look similar to the dataset image.

    Args:
        image_path: The file path where the image can be found.
    
    Returns:
        An image that has been preprocessed and is of size [1,784].

    Raises:
        If the file doesnt exist then tell the user you cant find the image.
     """
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