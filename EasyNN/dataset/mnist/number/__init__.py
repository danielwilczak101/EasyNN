from EasyNN.model import Network, Normalize, ReLU, LogSoftMax
from EasyNN.utilities.image.image import Image
from EasyNN.utilities.parameters.load import load
import matplotlib.pyplot as plt
import numpy as np

# Create the mnist model.
trained_model = Network(Normalize, 256, ReLU, 128, ReLU, 10, LogSoftMax)
# Set the parameters and stuff.
trained_model(np.empty(28 * 28))
trained_model.parameters = load("mnist_number_parameters.npy")
trained_model.layers[0]._mean = load("mnist_number_mean.npy")
trained_model.layers[0]._variance = load("mnist_number_variance.npy")
trained_model.layers[0]._weight = 1.0

# I will need the name of every veriable to load.
    # parameters => parameters
    # layer[0]._mean = > mean
    # layer[0]._variance => variance
    # layer[0]._weight => weight


# Set the dataset requirements
trained_model.training.file = "numbers.npz"
trained_model.training.url =  "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/number.npz"
trained_model.labels = {
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

# Image related functions.
def show(user_image: list[list[int]], image_type: str = None) -> None:
    """Show mnist 28 by 28 images as either array data or a matplotlib image.
    
    Args:
        user_image: User image to be shown or the path of the image is handled in the exceptions
        image_type: Tells whether to show the image as an image or print out.
            image: Show as matplotlib image. (default use.)
            array: Show as numpy formated array print out.

    Returns:
        Either an matplotlib graphed image or numpy print of the image pixel values.

    Example:
        >>> show(model.training.data[0])
        Shows plotted image of data point
        >>> show(model.training.data[0],"array")
        Shows print out of properly spaved numpy array data.
    """

    if image_type is None or image_type == "image":
        try:
            plt.imshow(user_image.reshape((28, 28)), cmap='gray')
            plt.show()
        except ValueError:
            raise ValueError(f"Image is not sized to [28,28] or [1,784].\n \
            Try: image = Image(user_image).format(resized=[28,28])") from None
    elif image_type == "array":
        # Show the array with corrected print width.
        np.set_printoptions(linewidth=114)
        # Show the image array data.
        print(user_image)
        # Return the print to the default 75.
        np.set_printoptions(linewidth=75)

trained_model.show = show