from __future__ import annotations
import numpy as np
from EasyNN.model.activation.abc import Activation
from EasyNN.loss.negative_likelihood import NegativeLikelihood
from EasyNN.typing import Array1D


class LogSoftMax(Activation):
    _default_loss = NegativeLikelihood

    def __backward__(self: LogSoftMax, dy: Array1D, y: Array1D = None, use_y: bool = False) -> Array1D:
        if not use_y:
            dy -= dy.sum(axis=-1, keepdims=True) * self.y
            return dy
        self.y = np.exp(self.y, out=self.y)
        if y.shape == dy.shape:
            self.y -= y
            return self.y
        elif y.ndim == 0:
            self.y[y] -= 1
            return self.y
        else:
            self.y[np.arange(len(y)), y] -= 1
            return self.y

    def __forward__(self: LogSoftMax, x: Array1D) -> Array1D:
        self.y = x.copy()
        self.y -= self.y.max(axis=-1, keepdims=True)
        self.y -= np.log(np.exp(self.y).sum(axis=-1, keepdims=True))
        return self.y
