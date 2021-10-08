"""
TODO: Not complete.
"""
from __future__ import annotations
from EasyNN.loss.abc import Loss
from EasyNN.typing import Array1D


class MeanSquareError(Loss):
    """
    Computes the loss as the mean squared error between the predicted
    output and the true output.
    """

    def __call__(self: MeanSquareError, x: Array1D, y: Array1D) -> float:
        y_pred = self.model(x)
        y_pred -= y
        y_pred **= 2
        return y_pred.mean() / 2

    def dy(self: MeanSquareError, y: Array1D, y_pred: Array1D) -> Array1D:
        y_pred -= y
        y_pred /= y.size
        return y_pred
