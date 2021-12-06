from EasyNN.examples.cifar10 import labels, show
from EasyNN.model import Network, Normalize, Randomize, ReLU, LogSoftMax
from EasyNN.optimizer import MomentumDescent
from EasyNN.batch import MiniBatch
import EasyNN.callbacks as cb


# Create the mnist model.
model = Network(
    Normalize(1e-6),
    Randomize(0.3), 1024, ReLU,
    Randomize(0.2), 256, ReLU,
    Randomize(0.1), 256, ReLU,
    Randomize(0.03), 10,  LogSoftMax
)

# Set the validation batch size.
model.validation.batch = MiniBatch(256)

# Establish the labels and show feature.
model.labels = labels
model.show = show

# Use gradient descent with momentum.
model.optimizer = MomentumDescent()

# Change the default learning rate.
model.optimizer.lr = 0.03

model.callback(
    # Set when to terminate point. 
        # In this case it will end once your validation accuracy hits above 90% five times.
    cb.ReachValidationAccuracy(limit=0.90, patience=2),
)

# When the model hit a validation point it will print the iteration and accuracy of the model.
model.print.on_validation_start(iteration=True, accuracy=True)
# When the model completes 10 iterations. It will print that iteration number.
model.print.on_training_start(iteration=True, frequency=10)
