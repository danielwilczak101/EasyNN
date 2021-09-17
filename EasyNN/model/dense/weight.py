from __future__ import annotations
from math import sqrt
from nptyping import NDArray
import numpy as np
from typing import Any, Union, overload
from EasyNN.model import Model
from EasyNN.typing import ArrayLike, Array1D, Array2D, Array3D, ArrayND, Shape1D, Shape2D, ShapeND


class Weight(Model):
    _shape_in: Shape1D = (Any,)
    _shape_out: Shape1D = (Any,)

    def __init__(self: Weight, matrix: ArrayLike = None, shape: Shape2D = None, neurons: Union[int, tuple[int]] = None) -> None:
        if (matrix, shape, neurons).count(None) < 2:
            raise TypeError("__init__() takes 2 arguments but {(matrix, shape, neurons).count(None) + 1} were given")
        elif matrix is not None:
            self.matrix = matrix
            self.setup()
        elif shape is not None:
            self.shape = shape
            self.setup()
        elif neurons is not None:
            self.shape_out = neurons if isinstance(neurons, tuple) else (neurons,)
        else:
            raise TypeError("__init__() missing 1 required argument: 'matrix', 'shape', or 'neurons'")

    @property
    def ndim(self: Weight) -> int:
        """Returns the number of dimensions of the weights matrix, which is 2."""
        return 2

    @property
    def shape(self: Weight) -> Shape2D:
        """The shape of the weights matrix, equivalent to (shape_out[0], shape_in[0])."""
        return self.shape_out + self.shape_in

    @shape.setter
    def shape(self: Weight, shape: Shape2D) -> None:
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError("shape must be a tuple of length 2")
        elif not all(s is Any or isinstance(s, int) for s in shape):
            raise ValueError("shape's elements must be Any or of type int")
        else:
            self.shape_in = shape[1:]
            self.shape_out = shape[:1]

    @property
    def shape_in(self: Weight) -> Shape1D:
        """The input shape is truncated to the last dimension, the rest are assumed batches."""
        return self._shape_in

    @shape_in.setter
    def shape_in(self: Weight, shape: ShapeND) -> None:
        self._shape_in = (shape[-1],)

    @property
    def shape_out(self: Weight) -> Shape1D:
        """The output shape is truncated to the last dimension, the rest are assumed batches."""
        return self._shape_out

    @shape_out.setter
    def shape_out(self: Weight, shape: ShapeND) -> None:
        self._shape_out = (shape[-1],)

    @property
    def size(self: Weight) -> Union[int, Literal[Any]]:
        """The size of the weights matrix, equivalent to shape[0] * shape[1]."""
        return Any if Any in self.shape else self.shape[0] * self.shape[1]

    @property
    def matrix(self: Weight) -> Array2D:
        return self.parameters.reshape(self.shape)

    @matrix.setter
    def matrix(self: Weight, matrix: ArrayLike) -> None:
        matrix = np.asarray(matrix, dtype=float)
        if not isinstance(matrix, Array2D):
            raise ValueError("matrix must be a 2D numpy array of floats")
        self.shape = matrix.shape
        self.parameters = matrix.reshape(-1)

    @overload
    def __backward__(self: Weight, dy: Array1D) -> Array1D:
        ...

    @overload
    def __backward__(self: Weight, dy: Array2D) -> Array2D:
        ...

    @overload
    def __backward__(self: Weight, dy: Array3D) -> Array3D:
        ...

    @overload
    def __backward__(self: Weight, dy: ArrayND) -> ArrayND:
        ...

    def __backward__(self, dy):
        if isinstance(dy, Array1D):
            self.derivatives = np.outer(dy, self.x)
            return self.matrix.T @ dy
        else:
            return self.__backward_batch__(derivatives)

    def __backward_batch__(self: Weight, dy: Array2D) -> Array2D:
        self.derivatives = derivatives @ self.x.T
        return self.matrix

    @overload
    def __forward__(self: Weight, x: Array1D) -> Array1D:
        ...

    @overload
    def __forward__(self: Weight, x: Array2D) -> Array2D:
        ...

    @overload
    def __forward__(self: Weight, x: Array3D) -> Array3D:
        ...

    @overload
    def __forward__(self: Weight, x: ArrayND) -> ArrayND:
        ...

    def __forward__(self, x):
        return self.matrix @ x

    def __setup__(self: Weight) -> None:
        if self.shape_in[0] is Any:
            self.shape_in = self.x.shape
        if not hasattr(self, "derivatives"):
            self.derivatives = np.empty(self.size)
        if not hasattr(self, "parameters"):
            std = sqrt(6 / sum(self.shape))
            self.parameters = np.random.uniform(-std, std, self.size)
