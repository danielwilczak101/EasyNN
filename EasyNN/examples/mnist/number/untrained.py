from EasyNN.examples.mnist.number.data import dataset
from EasyNN.examples.mnist.number import labels, show
from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.optimizer import MomentumDescent
from EasyNN.typing import Callback
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.download import download
from EasyNN.utilities.parameters import save
from EasyNN.utilities.momentum import Momentum
from EasyNN.batch import MiniBatch
import matplotlib.pyplot as plt
import numpy as np


# Create the mnist model.
model = Network(
    Normalize(1e-3),
    Randomize(0.5), 256, ReLU,
    Randomize(0.5), 256, ReLU,
    Randomize(0.3), 128, ReLU,
    Randomize(0.1), 10,  LogSoftMax
)

# Aim for 90% validation accuracy for 5 validation iterations in a row.
model.validation.accuracy_patience = 1
model.validation.accuracy_limit = 0.95
model.validation.successes = 0
model.validation_lr = 0.3
model.validation.accuracy = []
# Assign it some training/testing data.
model.training.data = dataset

# Extra features.
model.labels = labels
model.show = show

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

@model.on_optimization_start
def setup(model):
    model.validation.batch = MiniBatch(1024)

@model.on_validation_end
def terminate(model):
    if model.accuracy(*model.validation.sample) > model.validation.accuracy_limit:
        model.validation.successes += 1
    else:
        model.validation.successes = 0
    model.stop_training |= model.validation.successes >= model.validation.accuracy_patience



def printer(
    iteration: bool = False,
    training_loss: bool = False,
    training_accuracy: bool = False,
    validation_loss: bool = False,
    validation_accuracy: bool = False,
    frequency: int = 1,
) -> Callback:
    """Eventually probably a part of the utilities folder."""
    # Counter for the frequency.
    i = -1
    # Callback used by the model.
    def callback(model) -> None:
        # Increment the counter and if it doesn't match the frequency,
        # then just skip this callback.
        nonlocal i
        i += 1
        if i % frequency != 0:
            return
        # For the various variables, print the results.
        if iteration:
            print(f"{model.training.iteration = }")
        ...
    # Return the callback to be used with model.on_...(printer(...))
    return callback

# Print every 20 iterations.
model.on_training_iteration(printer(iteration=True, frequency=20))
# On each validation step, print the training loss and validation loss.
model.on_validation_start(printer(training_loss=True, validation_loss=True))
# On each validation step, print the training accuracy and validation accuracy.
model.on_validation_start(printer(training_accuracy=True, validation_accuracy=True))
# At the end during testing, check all of the losses.

@model.on_training_start
def print_training_iteration(model):
    if model.training.iteration % 10 == 0:
        print(f"  {model.training.iteration = }")

@model.on_validation_start
def print_validation_results(model):
    print(f"    {model.loss(*model.training.sample)   = }")
    print(f"    {model.loss(*model.validation.sample) = }")
    print(f"    {model.accuracy(*model.training.sample)   = }")
    print(f"    {model.accuracy(*model.validation.sample) = }")

    

@model.on_testing_start
def print_test_results(model):
    print(f"      {model.loss(*model.training.sample)   = }")
    print(f"      {model.loss(*model.validation.sample) = }")
    print(f"      {model.loss(*model.testing.sample)    = }")
    print(f"      {model.accuracy(*model.training.sample)   = }")
    print(f"      {model.accuracy(*model.validation.sample) = }")
    print(f"      {model.accuracy(*model.testing.sample)    = }")

@model.on_validation_start
def save_validation_accuracy(model):
    accuracy = model.accuracy(*model.validation.sample)
    model.validation.accuracy.append(accuracy)

@model.on_testing_start
def plot_validation(model):
    # Plot the validation accuracies.
    x = np.arange(model.validation.iteration + 1) * model.validation._batch_size / len(model.validation)
    y = model.validation.accuracy
    plt.plot(x, y, label="raw data")
    # Plot a smoothened version of the validation accuracies.
    smoothener = Momentum(0.3)
    y_smooth = [smoothener.update(accuracy) for accuracy in model.validation.accuracy]
    plt.plot(x, y_smooth, label="smoothened")
    # Setup the plot.
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(-0.1, 1.1)
    plt.legend(loc="lower right")
    plt.show()


# Once training has been finished save the model to use later
@model.on_testing_end
def save_data(model):
    save("number_model.npz", **model.get_arrays())

# Train the model and use the callback to get info and operate during training.
model.train()

# Download and example image.
file = "four.jpg"
url = "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/examples/four.jpg"
download(file, url)

# Establish formating options
format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

# Converting your image into the correct format for the mnist number dataset.
image = image(file).format(**format_options)

# Tell me what the image is.
print(model.classify(image))