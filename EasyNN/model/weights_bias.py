from __future__ import annotations
from typing import Optional, Tuple, Iterable
from numpy.typing import ArrayLike
import numpy as np
from EasyNN.math import row_vector, column_vector
from EasyNN.Model import Model


class Weights(Model):
    in_size: int
    out_size: int
    shape: Tuple[int, int]
    size: int
    matrix: np.ndarray

    def __init__(self: Weights, in_size: int, out_size: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        # only the output size is provided
        if out_size == 0:
            self.in_size = 0
            self.out_size = in_size
        # both the input and output sizes are provided
        else:
            self.in_size = in_size
            self.out_size = out_size
            self.parameters = np.empty((2, self.size))

    def pre_setup(self: Weights) -> None:
        # if the shape doesn't match up, the parameters need to be remade
        if self.in_size == self.inputs.shape[-1]:
            self.in_size = self.inputs.shape[-1]
            self.parameters = np.empty((2, self.size))

    def ff(self: Weights, values: np.ndarray) -> np.ndarray:
        return self.matrix @ values

    def ff_parallel(self: Weights, values: np.ndarray) -> np.ndarray:
        return (self.matrix @ values.T).T

    def bp(self: Weights, derivatives: np.ndarray) -> np.ndarray:
        self.derivatives = column_vector(derivatives) @ self.inputs.values
        return self.matrix.T @ derivatives

    def bp_parallel(self: Weights, derivatives: np.ndarray) -> np.ndarray:
        self.derivatives = derivatives.T @ self.inputs.values
        return self.matrix.T @ derivatives

    @property
    def is_parallel_inputs(self: Weights) -> bool:
        return self.inputs.ndim > 1

    @property
    def matrix(self: Weights) -> np.ndarray:
        return self.values.reshape(self.shape)

    @property
    def shape(self: Weights) -> Tuple[int, int]:
        return (self.out_size, self.in_size)

    @property
    def size(self: Weights) -> int:
        return self.out_size * self.in_size


class Bias(Model):
    dimensions: int
    shape: Tuple[int, ...] = ()
    shaped_values: np.ndarray

    def __init__(self: Bias, *, dimensions: int = 1, shape: Union[int, Iterable[int]] = (), **kwargs) -> None:
        # setup base model
        super().__init__(**kwargs)
        # if the shape is not an integer, it must be cast to tuple
        if not isinstance(shape, int):
            shape = tuple(shape)
        # shape is provided as an integer
        if isinstance(shape, int):
            # dimensions don't match shape
            if dimensions != 1:
                raise ValueError(f"inconsistent dimensions and shape, provided shape {shape} "
                                 f"has 1 dimensions, but {dimensions} dimensions was specified.")
            self.shape = (shape,)
            self.dimensions = 1
        # shape is provided as a tuple
        elif shape != ():
            # dimensions don't match shape
            if dimensions not in (1, len(shape)):
                raise ValueError(f"inconsistent dimensions and shape, provided shape {shape} "
                                 f"has {len(shape)} dimensions, but {dimensions} dimensions was specified.")
            self.shape = shape
            self.dimensions = len(shape)
        # shape is not provided
        else:
            self.shape = ()
            self.dimensions = dimensions

    def pre_setup(self: Bias) -> None:
        # check shape of the non-parallel dimensions
        input_shape = self.inputs.shape[-self.dimensions:]
        # if the shape doesn't match up, the parameters need to be remade
        if self.shape != input_shape:
            self.shape = input_shape
            self.parameters = np.empty((2, np.prod(input_shape)))

    def ff(self: Weights, values: np.ndarray) -> np.ndarray:
        return values + self.shaped_values

    def ff_parallel(self: Weights, values: np.ndarray) -> np.ndarray:
        return values + self.shaped_values

    def bp(self: Weights, derivatives: np.ndarray) -> np.ndarray:
        self.derivatives = derivatives
        return derivatives

    def bp_parallel(self: Weights, derivatives: np.ndarray) -> np.ndarray:
        # derivatives need to be summed along the parallel dimensions
        self.derivatives = derivatives.sum(axis=tuple(range(derivatives.ndim-self.dimensions)))
        return derivatives

    @property
    def is_parallel_inputs(self: Weights) -> bool:
        return self.inputs.ndim > self.dimensions

    @property
    def shaped_values(self: Weights) -> np.ndarray:
        return self.values.reshape(self.shape)
