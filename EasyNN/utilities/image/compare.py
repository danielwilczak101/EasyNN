import matplotlib.pyplot as plt
import random

__all__ = ["compare"]

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

        >>> compare(image, dataset = dataset)
        Returns subplot of users image and the rest fill in
        from the datasets images.
    """

    # Check if all the images are the same size.
        # If not raise - Images are not in the same format or size.

    # Maybe this should also be a model function to add simplicity to the function.
        # This is because each one changes its requirements based on the type of the image.

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
