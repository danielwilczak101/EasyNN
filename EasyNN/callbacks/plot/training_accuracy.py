from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
from typing import Literal

from EasyNN.callbacks.plot.abc import Plotter
from EasyNN.model.abc import Model


@dataclass
class PlotTrainingAccuracy(Plotter):
    """
    PlotTrainingAccuracy(
        lr: float = 0.3,
        name: str = "validation accuracy",
        x_axis: Literal["epochs", "iterations", "validation", "training"] = "epochs"
    )

    A Plotter callback for plotting the training accuracy.

    Parameters:
        lr:
            The rate at which the smoothened graph follows the raw data.
        name:
            The name of the data being plotted (for titles/labels).
        x_axis:
            Set the x-axis to epochs or iterations.
            By default, the epochs are used for the x-axis.
            By default, iterations is the validation iterations.
            May also be set to "training" for the training iterations.

    Example:
        Plot the training accuracy for the untrained mnist numbers model.
        >>> from EasyNN.examples.mnist.numbers.untrained import model
        >>> model.callback(PlotTrainingAccuracy())
        >>> model.train()
    """
    name: str = "Training accuracy"
    x_axis: Literal["epochs", "iterations", "validation", "training"] = "epochs"

    def x_values(self, model: Model) -> np.ndarray:
        if self.x_axis in ("iterations", "training"):
            return np.arange(len(self.data))
        elif self.x_axis == "validation":
            return np.arange(len(self.data)) * model.training._batch_size / model.validation._batch_size
        else:
            x = np.arange(len(self.data)) * model.training._batch_size / len(model.training)
            return x

    def on_validation_start(self, model: Model) -> None:
        self.data.append(model.accuracy(*model.training.sample))

    def on_testing_start(self, model: Model) -> None:
        super().on_testing_start(model)
        plt.xlabel(self.x_axis)
        plt.ylabel("accuracy")
        plt.ylim(-0.1, 1.1)