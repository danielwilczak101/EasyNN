from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.dataset.mnist.number import dataset
from EasyNN.batch import MiniBatch
from EasyNN.optimizer import MomentumDescent
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

# Create the cifar model.
model = Network(Normalize(1e-3), Randomize(0.5), 256, ReLU, Randomize(0.1), 128, ReLU, Randomize(0.1), 10, LogSoftMax)

# Assign it some training/testing data.
model.training.data = dataset

# Reduce the default learning rate.
model.optimizer = MomentumDescent()
model.optimizer.lr /= 2

#===================#
# Create callbacks: #
#===================#

@model.on_optimization_start
def callback(model):
    model.validation.batch = MiniBatch(1024)

@model.on_validation_start
def callback(model):
    if model.validation.iteration == 0:
        model._optimizer_lr /= 2 ** 0.5
    elif model.validation.iteration % 2 == 0:
        model._optimizer_lr /= 2
    else:
        model._optimizer_lr *= 2

@model.on_training_start
def callback(model):
    if model.training.iteration % 10 == 0:
        print(f"  {model.training.iteration = }")

@model.on_validation_start
def callback(model):
    print(f"  {model.loss(*model.training.sample)     = }")
    print(f"    {model.loss(*model.validation.sample) = }")
    print(f"  {model.accuracy(*model.training.sample)     = }")
    print(f"    {model.accuracy(*model.validation.sample) = }")

@model.on_testing_start
def callback(model):
    print(f"  {model.loss(*model.training.sample)     = }")
    print(f"    {model.loss(*model.validation.sample) = }")
    print(f"      {model.loss(*model.testing.sample)  = }")

model.validation_lr = 0.3
model.validation_losses = []
model.validation_smooth = []
model.validation_average = 0.0
model.validation_weight = 0.0

@model.on_validation_start
def callback(model):
    model.validation_losses.append(model.loss(*model.validation.sample))
    model.validation_weight -= model.validation_lr * (model.validation_weight - 1)
    model.validation_average -= model.validation_lr * (model.validation_average - model.validation_losses[-1])
    model.validation_smooth.append(model.validation_average / model.validation_weight)

@model.on_testing_start
def plot_validation(model):
    plt.plot(model.validation_losses)
    plt.plot(model.validation_smooth)
    plt.title("validation / iteration")
    plt.show()

def save_validation():
    np.save("validation_losses.npy", model.validation_losses)
    np.save("validation_smooth.npy", model.validation_smooth)

def load_validation():
    losses = np.load("validation_losses.npy")
    smooth = np.load("validation_smooth.npy")
    return losses, smooth

@model.on_validation_start
def callback(model):
    print("You have 5 seconds to use `[control] + C` to pause the program...")
    print("If you do, use `cont` to continue running or `quit()` to stop.")
    print("Use this to set `model.stop_training = True` to stop training.")
    sleep(5)

def main():
    print("Type `cont` to start:")
    breakpoint()
    prompt = "Type `cont` to start:"

    # Train the model.
    model.train()

    print("Any future things you might want to check out at the end of the program?")
    print("Type 'cont' to finish the program.")
    breakpoint()
    prompt = "Type 'cont' to finish the program."

if __name__ == "__main__":
    main()
