import numpy as np
from numpy.typing import ArrayLike
from EasyNN.model.activation.Activation import Activation


class Softmax(Activation):
    """The Softmax activation function."""

    def __call__(self, values: ArrayLike) -> np.ndarray:
        values = np.array(values, copy=False)

        # rescale the values so that overflow is avoided
        # exponentiate
        # divide by the sum of the powers
        if values.ndim == 1:
            self.powers = np.exp(values - values.max())
            self.powers /= self.powers.sum()
        else:
            self.powers = np.exp(values - values.max(axis=-1).reshape(-1, 1))
            self.powers /= self.powers.sum(axis=-1).reshape(-1, 1)

        return self.powers

    def backpropagate(self, derivatives: ArrayLike, loss: str = "") -> np.ndarray:

        if loss == "log likelihood":
            return derivatives

        elif derivatives.ndim == 1:
            result = self.powers * derivatives
            result -= self.powers * derivatives.sum()
            return result

        else:
            result = self.powers * derivatives
            result -= self.powers * derivatives.sum(axis=-1).reshape(-1, 1)
            return result
