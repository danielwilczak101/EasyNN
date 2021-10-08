"""
TODO: Documentation.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, overload
from EasyNN.model.abc import Model_1D
from EasyNN.typing import Array1D, Array2D


class Weight(Model_1D):
    """
    A Weight layer model, which matrix-multiplies its input by a matrix.

    Use Weight(out_neurons, /) to have the specified number of neurons as output.
    Use Weight(in_neurons, out_neurons, /) to specify the number of inputs as well.

    If only the out_neurons is specified, then the in_neurons is taken from the first input.
    Once the in_neurons is set, it cannot be changed.
    """
    neurons: int

    def __init__(self: Weight, *neurons: int) -> None:
        # Only allow integer inputs.
        if not all(isinstance(n, int) and n > 0 for n in neurons[:2]):
            raise TypeError("neurons must be positive integers")
        # Weight(10), collects the input into 10 neurons.
        elif len(neurons) == 1:
            self.neurons = neurons[0]
        # Weight(128, 10), collects 128 inputs into 10 neurons.
        elif len(neurons) == 2:
            self.x = np.empty(neurons[0])
            self.neurons = neurons[1]
            self.__setup__()
        # Invalid number of inputs.
        else:
            raise TypeError(f"__init__ expect 1-2 positional arguments but got {len(neurons)} instead")

    def __forward__(self: Weight, x: Array1D) -> Array1D:
        return x @ self.matrix.T

    def __backward__(self: Weight, dy: Array1D, y: Optional[Array1D] = None, use_y: bool = False) -> Array1D:
        if isinstance(dy, Array1D):
            self.derivatives = np.outer(dy, self.x)
        else:
            self.derivatives = dy.T @ self.x
            self.derivatives /= dy.shape[-1]
        return dy @ self.matrix

    def __setup__(self: Weight) -> None:
        if not hasattr(self, "parameters"):
            std = np.sqrt(6 / (self.neurons + self.x.shape[-1]))
            self.parameters = np.random.uniform(-std, std, self.neurons * self.x.shape[-1])
        if not hasattr(self, "derivatives"):
            self.derivatives = np.empty(self.neurons * self.x.shape[-1], dtype=float)

    @property
    def matrix(self: Weight) -> Array2D:
        return self.parameters.reshape(self.neurons, -1)

    @matrix.setter
    def matrix(self: Weight, matrix: Array2D) -> None:
        matrix = np.asarray(matrix, dtype=float)
        if not isinstance(matrix, (Array2D[float], Array2D[int])):
            raise ValueError("matrix must be a 2D array of floats or ints")
        self.parameters = matrix.flatten()
