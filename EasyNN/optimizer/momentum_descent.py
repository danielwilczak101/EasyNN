"""
TODO: Not complete.
"""
from __future__ import annotations
from typing import Any
from EasyNN.optimizer.gradient_descent import GradientDescent
import EasyNN.model.abc


class MomentumDescent(GradientDescent):
    """Momentum Descent uses parameters -= lr * momentum."""
    _momentum_lr: float = 3e-2

    def __init__(self: MomentumDescent, *args: Any, momentum_lr: float = 1e-2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._momentum_lr = momentum_lr

    def setup(self: MomentumDescent, model: EasyNN.model.abc.Model) -> None:
        """Setup a model."""
        super().setup(model)
        model._momentum_lr = self._momentum_lr
        model._momentum_weights = 0.0
        model._momentum_derivatives = 0.0

    def get_derivatives(self: Optimizer, model: EasyNN.model.abc.Model) -> Array1D[float]:
        """Computes the derivatives for the optimizer."""
        model._momentum_weights += model._momentum_lr * (1 - model._momentum_weights)
        model._momentum_derivatives += model._momentum_lr * (super().get_derivatives(model) - model._momentum_derivatives)
        return model._momentum_derivatives / model._momentum_weights
