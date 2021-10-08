import numpy as np
import matplotlib as plt

# Establish show and data.
def show(image):
    """Show the image as an RGB image.

        Args:
            image: Numpy array of an image structured in RGB and flattened.
        
        Return:
            image: [32,32] image of the RGB stacked on top of itself.

        Example:
            >>> show(xtrain[0])
            Will show a matplot lib of the a [32,32] image.
    """
    
    im_r = image[0:1024].reshape(32, 32)
    im_g = image[1024:2048].reshape(32, 32)
    im_b = image[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    plt.imshow(img) 
    plt.show()