"""
TODO: Not complete.
"""
from __future__ import annotations
from EasyNN.optimizer.abc import Optimizer
import EasyNN.model.abc


class GradientDescent(Optimizer):
    """Gradient Descent uses parameters -= lr * derivatives."""

    def on_training_start(self: GradientDescent, model: EasyNN.model.abc.Model) -> None:
        model.sample_derivatives(*model.training.sample)
        model.parameters -= self.lr * model.derivatives
