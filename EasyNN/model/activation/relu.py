from __future__ import annotations
import numpy as np
from typing import Optional, overload
from EasyNN.model.activation.abc import Activation
from EasyNN.typing import Array1D, Array2D, Array3D, ArrayND


class ReLU(Activation):

    @overload
    def __backward__(self: ReLU, dy: Array1D, y: Optional[Array1D] = ..., use_y: bool = ...) -> Array1D:
        ...

    @overload
    def __backward__(self: ReLU, dy: Array2D, y: Optional[Array2D] = ..., use_y: bool = ...) -> Array2D:
        ...

    def __backward__(self, dy, y=None, use_y=False):
        dy *= np.sign(self.x)
        return dy

    @overload
    def __forward__(self: ReLU, x: Array1D) -> Array1D:
        ...

    @overload
    def __forward__(self: ReLU, x: Array2D) -> Array2D:
        ...

    def __forward__(self, x):
        return np.maximum(x, 0, out=x)
