from EasyNN.utilities.parameters import save
from examples.mnist.number.untrained import model

# Once training has been finished save the model to use later
@model.on_testing_end
def save_data(model):
    save("model.npz", **model.get_arrays())

# Train the model.
model.train()


