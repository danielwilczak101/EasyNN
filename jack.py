from EasyNN.model import Network, ReLU, LogSoftMax
from EasyNN.dataset.mnist.fashion import dataset

# Create the mnist model.
model = Network(128, ReLU, 128, ReLU, 10, LogSoftMax)

# Assign it some training/testing data.
model.training.data = dataset

# Reduce the default learning rate.
model.optimizer.lr /= 3
#===================#
# Create callbacks. #
#===================#

@model.on_training_start
def callback():
    print(f"  {model.training.iteration = }")

@model.on_training_end
def callback():
    model.stop_training |= model.training.iteration == 99

@model.on_validation_start
def callback():
    print(f"  {model.loss(*model.training.sample) = }")
    print(f"    {model.loss(*model.validation.sample) = }")

@model.on_testing_start
def callback():
    print(f"  {model.loss(*model.training.sample) = }")
    print(f"    {model.loss(*model.validation.sample) = }")
    print(f"      {model.loss(*model.testing.sample) = }")

# Train the model.
model.train()
