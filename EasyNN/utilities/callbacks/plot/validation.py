import matplotlib.pyplot as plt
import numpy as np
from EasyNN.utilities.momentum import Momentum

def accuracy(model):
    # Plot the validation accuracies.
    x = np.arange(model.validation.iteration + 1) * model.validation._batch_size / len(model.validation)
    y = model.validation.accuracy
    plt.plot(x, y, label="raw data")
    # Plot a smoothened version of the validation accuracies.
    smoothener = Momentum(0.3)
    y_smooth = [smoothener.update(accuracy) for accuracy in model.validation.accuracy]
    plt.plot(x, y_smooth, label="smoothened")
    # Setup the plot.
    plt.yscale = "log"
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(-0.1, 1.1)
    plt.legend(loc="lower right")
    plt.show()
