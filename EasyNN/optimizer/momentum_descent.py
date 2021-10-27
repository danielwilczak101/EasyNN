"""
TODO: Not complete.
"""
from __future__ import annotations
from typing import Any
from EasyNN.optimizer.gradient_descent import GradientDescent
from EasyNN.utilities.momentum import Momentum
import EasyNN.model.abc


class MomentumDescent(GradientDescent):
    """Momentum Descent uses parameters -= lr * momentum."""
    _momentum_lr: float

    def __init__(self: MomentumDescent, *args: Any, momentum_lr: float = 1e-2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._momentum_lr = momentum_lr

    def setup(self: MomentumDescent, model: EasyNN.model.abc.Model) -> None:
        """Setup a model."""
        super().setup(model)
        model._derivative_momentum = Momentum(self._momentum_lr)

    def get_derivatives(self: MomentumDescent, model: EasyNN.model.abc.Model) -> Array1D[float]:
        """Computes the derivatives for the optimizer."""
        if model.command.startswith("on_training"):
            return model._derivative_momentum.update(super().get_derivatives(model))
        else:
            return model._derivative_momentum.peek(super().get_derivatives(model))
