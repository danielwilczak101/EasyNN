from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.dataset.mnist.number import dataset
from EasyNN.batch import MiniBatch
from EasyNN.optimizer import MomentumDescent
from EasyNN.utilities.momentum import Momentum
from EasyNN.utilities.image.image import Image
from EasyNN.utilities.download import download
from EasyNN.utilities.parameters.save import save
import matplotlib.pyplot as plt
import numpy as np

#==================#
# Setup the model: #
#==================#

# Create the mnist model.
model = Network(
    Normalize(1e-3),
    Randomize(0.5),
    256, ReLU,
    Randomize(0.3),
    128, ReLU,
    Randomize(0.1),
    10, LogSoftMax
)

# Assign it some training/testing data.
model.training.data = dataset
model.labels = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
}

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

# Aim for 90% validation accuracy for 5 validation iterations in a row.
model.validation.accuracy_patience = 5
model.validation.accuracy_limit = 0.90
model.validation.successes = 0

#===================#
# Create callbacks: #
#===================#

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

    save(model.parameters, "mnist_number_parameters.npy")
    save(model.mean, "mnist_number_parameters.npy")
    save(model.variance, "mnist_number_parameters.npy")

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

model.validation_lr = 0.3
model.validation.accuracy = []

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

#====================================#
# Main code if file is run directly. #
#====================================#

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

file = "four.jpg"
url = "https://github.com/danielwilczak101/EasyNN/raw/datasets/mnist/number/examples/four.jpg"

def main():
    """Simply run `model.train()` to train the model."""

    # Train the model.
    model.train()
        
    # Download an example image.
    download(file, url)

    # Converting your image into the correct format for the mnist number dataset.
    image = Image(file).format(**format_options)

    print(model.classify(image))
