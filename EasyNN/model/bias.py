"""
TODO: Documentation.
"""
from __future__ import annotations
import numpy as np
from typing import overload
from EasyNN.model.abc import Model_1D
from EasyNN.typing import Array1D


class Bias(Model_1D):
    neurons: int

    def __init__(self: Bias, neurons: int = None) -> None:
        if neurons is not None:
            self.neurons = neurons

    def __backward__(self: Bias, dy: Array1D, y: Array1D = None, use_y: bool = False) -> Array1D:
        if dy.shape != self.x.shape:
            raise ValueError(f"expected the same shape as the input, which is {self.x.shape}, not {dy.shape}")
        elif dy.ndim == 1:
            self.derivatives = dy
        elif dy.ndim == 2:
            self.derivatives = dy.mean(axis=0)
        else:
            raise ValueError(f"expected either 1-D or 2-D arrays, not {dy.ndim} arrays")
        return dy

    def __forward__(self: Bias, x: Array1D) -> Array1D:
        if x.ndim not in (1, 2):
            raise ValueError(f"expected either a 1-D or 2-D arrays, not a {dy.ndim}-D array")
        x += self.parameters
        return x

    def __setup__(self: Bias) -> None:
        # Get the number of neurons from the input.
        if not hasattr(self, "neurons"):
            self.neurons = self.x.shape[-1]
        # Setup the parameters and derivatives using the number of neurons.
        if not hasattr(self, "parameters"):
            self.parameters = np.zeros(self.neurons, dtype=float)
        if not hasattr(self, "derivatives"):
            self.derivatives = np.empty(self.neurons, dtype=float)
