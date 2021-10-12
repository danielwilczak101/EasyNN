"""
TODO: Not complete.
"""
from __future__ import annotations
import numpy as np
from typing import Any
from EasyNN.optimizer.abc import Optimizer
from EasyNN.optimizer.momentum_descent import MomentumDescent
from EasyNN.utilities.momentum import Momentum
import EasyNN.model.abc


class Adam(MomentumDescent):
    """Adam optimizer uses parameters -= lr * momentum / L2_norm."""
    _L2_lr: float
    _L2_epsilon: float

    def __init__(self: MomentumDescent, *args: Any, L2_lr: float = 1e-3, L2_epsilon: float = 1e-7, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._L2_lr = L2_lr
        self._L2_epsilon = L2_epsilon

    def setup(self: MomentumDescent, model: EasyNN.model.abc.Model) -> None:
        """Setup a model."""
        super().setup(model)
        model._L2_momentum = Momentum(self._L2_lr)
        model._L2_epsilon = self._L2_epsilon

    def get_derivatives(self: MomentumDescent, model: EasyNN.model.abc.Model) -> Array1D[float]:
        """Computes the derivatives for the optimizer."""
        derivatives = Optimizer.get_derivatives(self, model)
        return model._derivative_momentum.update(derivatives) / np.sqrt(model._L2_momentum.update(derivatives ** 2) + model._L2_epsilon)
