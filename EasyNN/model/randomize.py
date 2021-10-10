from __future__ import annotations
import numpy as np
from typing import Generic, TypeVar
from EasyNN.model.abc import Model
from EasyNN.typing import ArrayND

ArrayShape = TypeVar("ArrayShape", bound=ArrayND)


class Randomize(Model[ArrayShape, ArrayShape], Generic[ArrayShape]):
    """Randomizes the training input for data augmentation purposes."""
    _parameters: ArrayND = np.empty(0, dtype=float)
    _derivatives: ArrayND = np.empty(0, dtype=float)
    noise: float

    def __init__(self: Randomize[ArrayShape], noise: float = 0.1) -> None:
        self.noise = noise

    def __forward__(self: Randomize[ArrayShape], x: ArrayShape) -> ArrayShape:
        if self.command.startswith("on_training_"):
            return x + np.random.normal(scale=self.noise, size=x.shape)
        return x

    def __backward__(self: Randomize[ArrayShape], dy: ArrayShape, y: ArrayShape = None, use_y: bool = False) -> ArrayShape:
        return dy

    def __setup__(self: Normalize[ArrayShape]) -> None:
        pass
