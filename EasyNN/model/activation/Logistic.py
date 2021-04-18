import numpy as np
from numpy.typing import ArrayLike
from EasyNN.model.activation.Activation import Activation


class Logistic(Activation):
    """The logistic function."""

    def __call__(self, values: ArrayLike) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-np.array(values, copy=False)))
        return self.output

    def backpropagate(self, derivatives: ArrayLike) -> np.ndarray:
        return self.output * (1 - self.output) * derivatives
