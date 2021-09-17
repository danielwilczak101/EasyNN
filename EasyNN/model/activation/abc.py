from __future__ import annotations
from abc import ABC
import numpy as np
from typing import Union, overload
from EasyNN.model.abc import Model
from EasyNN.typing import Array1D, Array2D, Array3D, ArrayND


class Activation(Model, ABC):
    """
    Activation Models typically involve vectorized
    functions applied over the last ndim dimensions.

    By default, only the last dimension is operated on
    and the remaining dimensions are assumed to be a batch.
    """
    _ndim: int = 1

    @property
    def ndim(self: Activation) -> int:
        """
        The number of dimensions being applied to,
        the rest are assumed batch dimensions.
        """
        return self._ndim

    @ndim.setter
    def ndim(self: Activation, ndim: int) -> None:
        if not isinstance(ndim, int):
            raise TypeError(f"ndim must be an int, not {type(ndim).__name__}")
        elif ndim <= 0:
            raise ValueError(f"ndim must be an int > 0")
        self._ndim = ndim

    @property
    def axis(self: Activation) -> tuple[int, ...]:
        """The last ndim axes which are vectorized over."""
        return tuple(range(-1, -self.ndim-1, -1))

    def __setup__(self: Activation) -> None:
        """Activations usually require no parameters."""
        self.parameters = np.empty(0, dtype=float)
        self.derivatives = np.empty(0, dtype=float)
