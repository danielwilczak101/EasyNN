"""
TODO: Not complete.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import partial
from itertools import count, cycle
from typing import get_args
from EasyNN._abc import AutoDocumentation
from EasyNN.typing import Command
import EasyNN.model.abc


class Optimizer(AutoDocumentation, ABC):
    """
    Abstract Base Class for the Optimizer.
    """
    models: dict[Hashable, EasyNN.model.abc.Model]
    lr: float

    def __init__(self: Optimizer, models: dict[Hashable, EasyNN.model.abc.Model] = None, lr: float = 1e-2) -> None:
        self.models = dict() if models is None else models
        self.lr = lr

    def add(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Add a model to the optimizer."""
        for i in count(len(self.models)):
            if i not in self.models:
                self.models[i] = model
                break

    def train(self: Optimizer, *models: EasyNN.model.abc.Model) -> tuple[EasyNN.model.abc.Model, ...]:
        """
        Train the models.

        Example:
            >>> optimizer.train(model_1, model_2)
            (model_1, model_2)
        """
        # By default, use the saved optimizer models.
        if len(models) == 0:
            models = tuple(self.models.values())
        # Setup the model commands.
        for model in models:
            # Add on the optimizer commands.
            for command in get_args(Command):
                model.callback(command)(partial(getattr(self, command), model))
            # Begin running model commands.
            model.commands = model.optimizer_commands()
        # Cycle through each model.
        for model in cycle(models):
            # Run the next command for the model and stop if there are no more.
            if next(model.commands, None) is None:
                return models

    def on_optimization_start(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Ran at the start of optimization."""

    def on_optimization_end(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Ran at the end of optimization."""

    @abstractmethod
    def on_training_start(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Ran at the start of every training iteration. Required for optimizers."""
        raise NotImplementedError

    def on_training_end(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Ran at the end of every training iteration."""

    def on_testing_start(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Ran at the start of every testing iteration. This is initialized after """

    def on_testing_end(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Ran at the end of every testing iteration."""

    def on_validation_start(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Ran at the start of every validation iteration."""

    def on_validation_end(self: Optimizer, model: EasyNN.model.abc.Model) -> None:
        """Ran at the end of every validation iteration."""

    def on_epoch_start(self: Optimizer, model: EasyNN.abc.Model) -> None:
        """Ran at the start of every epoch."""

    def on_epoch_end(self: Optimizer, model: EasyNN.abc.Model) -> None:
        """Ran at the end of every epoch."""
