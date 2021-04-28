import numpy as np
from numpy.typing import ArrayLike
from EasyNN.model.activation.Activation import Activation
from EasyNN.model.Model import Model


class ReLU(Activation, Model):
    """The Rectified Linear Unit."""

    def __call__(self, values: ArrayLike) -> np.ndarray:
        self.input = np.array(values, copy=False)
        return np.maximum(values, 0)

    def backpropagate(self, derivatives: ArrayLike) -> np.ndarray:
        return np.where(self.input > 0, derivatives, 0)
