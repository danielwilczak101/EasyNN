from __future__ import annotations
import numpy as np
from typing import overload
from EasyNN.model.activation.abc import Activation
from EasyNN.typing import Array1D, Array2D, Array3D, ArrayND


class ReLU(Activation):

    @overload
    def __backward__(self: ReLU, dy: Array1D) -> Array1D:
        ...

    @overload
    def __backward__(self: ReLU, dy: Array2D) -> Array2D:
        ...

    @overload
    def __backward__(self: ReLU, dy: Array3D) -> Array3D:
        ...

    @overload
    def __backward__(self: ReLU, dy: ArrayND) -> ArrayND:
        ...

    def __backward__(self, dy):
        dy *= np.sign(self.x)
        return dy

    @overload
    def __forward__(self: ReLU, x: Array1D) -> Array1D:
        ...

    @overload
    def __forward__(self: ReLU, x: Array2D) -> Array2D:
        ...

    @overload
    def __forward__(self: ReLU, x: Array3D) -> Array3D:
        ...

    @overload
    def __forward__(self: ReLU, x: ArrayND) -> ArrayND:
        ...

    def __forward__(self, x):
        return np.maximum(x, 0, out=x)
