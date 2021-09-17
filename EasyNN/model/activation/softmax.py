from __future__ import annotations
import numpy as np
from typing import overload
from EasyNN.model.activation.abc import Activation
from EasyNN.typing import Array1D, Array2D, Array3D, ArrayND


class SoftMax(Activation):
    ndim: int
    alpha: float
    y: ArrayND

    def __init__(self: SoftMax, ndim: int = 1, alpha: float = 1.0) -> None:
        self.ndim = ndim
        self.alpha = alpha

    @overload
    def __backward__(self: SoftMax, dy: Array1D) -> Array1D:
        ...

    @overload
    def __backward__(self: SoftMax, dy: Array2D) -> Array2D:
        ...

    @overload
    def __backward__(self: SoftMax, dy: Array3D) -> Array3D:
        ...

    @overload
    def __backward__(self: SoftMax, dy: ArrayND) -> ArrayND:
        ...

    def __backward__(self, dy):
        dy *= self.y
        dy -= dy.sum(axis=self.axis, keepdims=True) * self.y
        dy *= self.alpha
        return dy

    @overload
    def __forward__(self: SoftMax, x: Array1D) -> Array1D:
        ...

    @overload
    def __forward__(self: SoftMax, x: Array2D) -> Array2D:
        ...

    @overload
    def __forward__(self: SoftMax, x: Array3D) -> Array3D:
        ...

    @overload
    def __forward__(self: SoftMax, x: ArrayND) -> ArrayND:
        ...

    def __forward__(self, x):
        x -= x.max(axis=self.axis, keepdims=True)
        x *= self.alpha
        x = np.exp(x, out=x)
        x /= x.sum(axis=self.axis, keepdims=True)
        # Store the y value instead of the x value for backpropagation.
        del self.x
        self.y = x
        return x
