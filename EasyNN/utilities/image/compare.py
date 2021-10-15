import matplotlib.pyplot as plt

def compare(*args,width=4):
    """
    Used to compare matplotlib images to eachother.

    Args:
        *args: All the images given to the function
        width: Used to tell the maximum images you want on one row.

    Return:
        Subplot of all the images.

    Example:
        >>> compare(image1,image2,image3)
        Returns subplot of image1,image2,image3
    """
    # Not finished yet.
    for image in args:
        
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)