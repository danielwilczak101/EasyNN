from __future__ import annotations
from math import sqrt, prod
import numpy as np
from typing import Any, Generic, Literal, TypeVar, Union, overload
from EasyNN.model import Model
from EasyNN.typing import ArrayLike1D, ArrayLike2D, ArrayLike3D, ArrayLikeND, Array1D, Array2D, Array3D, ArrayND, Shape1D, Shape2D, ShapeND

ShapeVar = TypeVar("ShapeVar", bound=ShapeND)


class Bias(Generic[ShapeVar], Model):
    _ndim: int
    _shape: ShapeVar

    @overload
    def __init__(self: Bias[Shape1D], shape: int) -> None:
        ...

    @overload
    def __init__(self: Bias[ShapeVar], shape: ShapeVar) -> None:
        ...

    @overload
    def __init__(self: Bias[Shape1D], *, parameters: ArrayLike1D) -> None:
        ...

    @overload
    def __init__(self: Bias[Shape2D], *, parameters: ArrayLike2D) -> None:
        ...

    @overload
    def __init__(self: Bias[Shape3D], *, parameters: ArrayLike3D) -> None:
        ...

    @overload
    def __init__(self: Bias[ShapeVar], *, parameters: ArrayLikeND) -> None:
        ...

    @overload
    def __init__(self: Bias[Shape1D], *, ndim: Literal[1] = ...) -> None:
        ...

    @overload
    def __init__(self: Bias[Shape2D], *, ndim: Literal[2]) -> None:
        ...

    @overload
    def __init__(self: Bias[Shape3D], *, ndim: Literal[3]) -> None:
        ...

    @overload
    def __init__(self: Bias[ShapeVar], *, ndim: int) -> None:
        ...

    def __init__(self, shape=None, *, parameters=None, ndim=1):
        if shape is not None:
            self.ndim = 1 if isinstance(shape, int) else len(shape)
            self.shape = shape
            self.setup()
        elif parameters is not None:
            self.parameters = parameters
            self.shape = self.parameters.shape
            self.setup()
        else:
            self.ndim = ndim

    @property
    def ndim(self: Bias[ShapeVar]) -> int:
        """The number of dimensions of the bias parameters, equivalent to len(self.shape)."""
        return self._ndim

    @ndim.setter
    def ndim(self: Bias[ShapeVar], ndim: int) -> None:
        if not isinstance(ndim, int):
            raise TypeError(f"ndim must be an int, not {type(ndim).__name__}")
        elif ndim <= 0:
            raise ValueError(f"ndim must be an int > 0")
        self._ndim = ndim

    @property
    def shape(self: Bias[ShapeVar]) -> ShapeVar:
        """The shape of the bias parameters, equivalent to self.shape_in."""
        return self._shape

    @shape.setter
    def shape(self: Bias[ShapeVar], shape: Union[ShapeVar, int]) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        elif not isinstance(shape, tuple):
            raise TypeError("shape must be an int or tuple")
        elif not all(s is Any or isinstance(s, int) for s in shape):
            raise TypeError("shape tuples may only contain Any or an int e.g. (Any, 3)")
        if not hasattr(self, "ndim"):
            self.ndim = len(shape)
        self._shape = shape[-self.ndim:]

    @property
    def shape_in(self: Bias[ShapeVar]) -> ShapeVar:
        """The input shape is truncated to the last ndim dimensions, the rest are assumed batches."""
        return self.shape

    @shape_in.setter
    def shape_in(self: Bias[ShapeVar], shape: Union[ShapeVar, int]) -> None:
        self.shape = shape

    @property
    def shape_out(self: Bias[ShapeVar]) -> ShapeVar:
        """The output shape is truncated to the last ndim dimensions, the rest are assumed batches."""
        return self.shape

    @shape_out.setter
    def shape_out(self: Bias[ShapeVar], shape: Union[ShapeVar, int]) -> None:
        self.shape = shape

    @property
    def size(self: Bias[ShapeVar]) -> Union[int, Literal[Any]]:
        """The size of the bias parameters, equivalent to (shape[0] * shape[1] * ...)."""
        return Any if Any in self.shape else np.prod(self.shape)

    @overload
    def __backward__(self: Bias[Shape1D], dy: Array1D) -> Array1D:
        ...

    @overload
    def __backward__(self: Bias[Shape2D], dy: Array2D) -> Array2D:
        ...

    @overload
    def __backward__(self: Bias[Shape3D], dy: Array3D) -> Array3D:
        ...

    @overload
    def __backward__(self: Bias[ShapeVar], dy: ArrayND) -> ArrayND:
        ...

    def __backward__(self, dy):
        if dy.shape != self.x.shape:
            raise ValueError(f"expected the same shape as the input, which is {self.x.shape}, not {dy.shape}")
        elif dy.ndim > self.ndim:
            return self.__backward_batch__(dy)
        self.dy = dy
        return dy

    @overload
    def __backward_batch__(self: Bias[Shape2D], dy: Array2D) -> Array2D:
        ...

    @overload
    def __backward_batch__(self: Bias[Shape3D], dy: Array3D) -> Array3D:
        ...

    @overload
    def __backward_batch__(self: Bias[ShapeVar], dy: ArrayND) -> ArrayND:
        ...

    def __backward_batch__(self, dy):
        self.derivatives = dy.mean(axis=tuple(range(len(dy) - self.ndim)))
        return dy

    def __forward__(self: Bias[ShapeVar], x: ArrayND) -> ArrayND:
        x += self.parameters
        return x

    def __setup__(self: Bias[ShapeVar]) -> None:
        if not hasattr(self, "shape"):
            self.shape = self.x.shape
        if not hasattr(self, "derivatives"):
            self.derivatives = np.empty(self.shape, dtype=float)
        if not hasattr(self, "parameters"):
            self.parameters = np.zeros(self.shape, dtype=float)
