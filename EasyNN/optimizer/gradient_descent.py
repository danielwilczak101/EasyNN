"""
TODO: Not complete.
"""
from __future__ import annotations
from EasyNN.optimizer.abc import Optimizer
import EasyNN.model.abc


class GradientDescent(Optimizer):
    """Gradient Descent uses parameters -= lr * derivatives."""

    def on_training_start(self: GradientDescent, model: EasyNN.model.abc.Model) -> None:
        model.parameters -= model._optimizer_lr / (1 + 1e-2 * model.training.iteration) ** 0.3 * self.get_derivatives(model)
