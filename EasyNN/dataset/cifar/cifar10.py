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
data_filename  = "cifar10.npz"
model_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/cifar/cifar10.model"
data_url  = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/cifar/cifar10.npz"

labels = {
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

# Establish structure.
model = Network()
model.add(Layer_Dense(3072, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 10))
model.add(Activation_Softmax())
model.set(
    # Set loss, optimizer and accuracy objects
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
    )

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
# Give the models the labels.
model.show = show
model.labels = labels
# Finialize everything.
model.finalize()

def __getattr__(name):
    if name == "model": 
        untrained_model = make_model(
            model,
            model_filename,
            data_filename,
            data_url,
            epochs=1000
        )      
        return untrained_model

    if name == "trained_model":
        trained_model = make_trained_model(
        model,
        model_filename,
        model_url,
        data_filename,
        data_url
    )
        return trained_model
        
    raise AttributeError(f"module {__name__} has no attribute {name}")