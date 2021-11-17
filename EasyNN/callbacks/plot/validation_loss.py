from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
from typing import Literal

from EasyNN.callbacks.plot.abc import Plotter
#from EasyNN.model.abc import Model


@dataclass
class PlotValidationLoss(Plotter):
    """
    PlotValidationLoss(
        lr: float = 0.3,
        name: str = "validation loss",
        x_axis: Literal["epochs", "iterations", "validation", "training"] = "epochs"
    )

    A Plotter callback for plotting the validation loss.

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
        Plot the validation loss for the untrained mnist numbers model.
        >>> from EasyNN.examples.mnist.numbers.untrained import model
        >>> model.callback(PlotValidationLoss())
        >>> model.train()
    """
    name: str = "validation loss"
    x_axis: Literal["epochs", "iterations", "validation", "training"] = "epochs"

    def x_values(self, model) -> np.ndarray:
        if self.x_axis in ("iterations", "validation"):
            return np.arange(len(self.data))
        elif self.x_axis == "training":
            return np.arange(len(self.data)) * model.validation._batch_size / model.training._batch_size
        else:
            return np.arange(len(self.data)) * model.validation._batch_size / len(model.validation)

    def on_validation_start(self, model) -> None:
        self.data.append(model.loss(*model.validation.sample))

    def on_testing_start(self, model) -> None:
        super().on_testing_start(model)
        plt.xlabel(self.x_axis)
        plt.ylabel("loss")