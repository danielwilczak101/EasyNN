from __future__ import annotations
from abc import ABC
import numpy as np
from EasyNN.model.abc import Model_1D
from EasyNN.typing import Array1D


class Activation(Model_1D, ABC):
    """
    Activation Models typically involve vectorized
    functions applied over the last dimension.
    """
    _parameters: Array1D = np.empty(0, dtype=float)
    _derivatives: Array1D = np.empty(0, dtype=float)

    def __setup__(self: Activation) -> None:
        pass
