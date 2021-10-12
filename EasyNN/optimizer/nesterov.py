"""
TODO: Not complete.
"""
from __future__ import annotations
from EasyNN.optimizer.momentum_descent import MomentumDescent
import EasyNN.model.abc


class Nesterov(MomentumDescent):
    """Nesterov uses parameters -= lr * momentum, where the momentum is computed by looking ahead."""

    def get_derivatives(self: Nesterov, model: EasyNN.model.abc.Model) -> Array1D[float]:
        """Computes the derivatives for the optimizer."""
        if model.training.iteration == 0:
            return super().get_derivatives(model)
        parameters = model.parameters.copy()
        self.on_training_start(model, model._derivative_momentum.value)
        super().get_derivatives(model)
        model.parameters = parameters
        return model._derivative_momentum.value
