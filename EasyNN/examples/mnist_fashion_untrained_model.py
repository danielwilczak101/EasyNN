from EasyNN.examples.mnist.fashion.untrained import model
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.download import download
from EasyNN.utilities.parameters import save
from EasyNN.utilities.momentum import Momentum
from EasyNN.batch import MiniBatch
import matplotlib.pyplot as plt
import numpy as np


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


# Once training has been finished save the model to use later
@model.on_testing_end
def save_data(model):
    save("fashion_model.npz", **model.get_arrays())

# Train the model and use the callback to get info and operate during training.
model.train()