from EasyNN.examples.mnist.fashion.untrained import model
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.download import download
from EasyNN.utilities.parameters import save
from EasyNN.examples.mnist.number.data import dataset
from EasyNN.examples.mnist.number import show

# Once training has been finished save the model to use later
@model.on_testing_end
def save_data(model):
    save("fashion_model.npz", **model.get_arrays())
    
# Train the model.
model.train()
images, labels = dataset
print(model.accuracy(images, labels))
