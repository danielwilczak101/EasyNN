from EasyNN.utilities import Preprocess
import numpy as np
import os

def create_image_dataset(name:str,path:str,size:list[int]=(28,28),greyscale:bool=True):
    """ 
    Create a dataset from images in a folders with folder names as labels.

    Example:
        Images:
            ├── cat <- Folder and label to the images.
            │   ├── 0.jpg <- Image of cat
            │   ├── 1.jpg
            └── dog <- Folder and label to the images.
                ├── 0.jpg <- Image of dog
                ├── 1.jpg <- Image of dog

    >>> create_image_dataset("Animal", "./images/", (28,28))
    >>> dataset = load("Animal_dataset.npz")
    >>> images, labels = dataset
    >>> print(images.shape)
    >>> print(labels.shape)
    (4, 784)
    (4,)

    Shows 4 labels and 4 images to be used in a neural network.

    Arguements:
        name: Name of the of dataset file the data will be saved to.
        path: Path to the folder where your labeled folders are.
        size: Size the images will be resized to.
            Default = (28,28) pixels
        greyscale: Change the image to greyscale values.
            Default = True

    Returns:
        Numpy file with images and labels stored in numpy arrays.
    """
    x = np.zeros(shape = (1, 784), dtype = int)
    y = np.array([], dtype = int)
    labels = np.array([], dtype= str)

    # Each folder in the directory.
    for index,folder in enumerate(os.listdir(path)):

        # List all the files in the folder selected.
        files = os.listdir(path+folder)
        files:list[str]

        # The label is the folder name.
        folder:str

        # Store the folder name into the labels array.
        labels = np.append(labels,folder)

        for file in files:
            format_options = dict(
                grayscale=True,
                resize=size
            )

            # Converting your image into the correct format for the mnist number dataset.
            image = Preprocess(path+folder+'/'+file).format(**format_options)

            x = np.append(x,[image],axis = 0)
            y = np.append(y,index)

    # Removes the first empty array.
    x = np.delete(x,0,0)
    # Save the file.
    np.savez(name+"_dataset", x=x, y=y)