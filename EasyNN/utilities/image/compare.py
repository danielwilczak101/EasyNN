import matplotlib.pyplot as plt
import random

def compare(*args, width=3, height=3, dataset=None) -> None:
    """
    Used to compare matplotlib images to eachother.

    Args:
        *args: All the images given to the function
        width: Used to tell the maximum images you want on one row.
        dataset(x,y): If the user wants to compare their image
        to a bunch of dataset images they can feed in the whole
        dataset and the remaing image spot will be filled. 

    Return:
        Subplot of all the images.

    Example:
        >>> compare(image1,image2,image3)
        Returns subplot of image1,image2,image3
    """

        # First images are the user defined ones.
            # If the dataset variable is set we fill in using dataset.
        
        # The maximum images we use is set by width.

        # Each image take up 110pixels width on the plot.

        # Put the label above the images.
            # User definded one is not required.

    args_count = len(args)

    for i in range(args_count):
        # define subplot
        plt.subplot(width * 110 + 1 + i)
        # plot raw pixel data
        plt.imshow(args[i].reshape([28, 28]), cmap=plt.get_cmap('gray'))
    
    if dataset:
        for i in range(args_count, width * height):
            # define subplot
            plt.subplot(width * 110 + 1 + i)
            # plot raw pixel data
            plt.imshow(random.choice(dataset[0]).reshape([28, 28]), cmap=plt.get_cmap('gray'))

    # show the figure
    plt.show()
