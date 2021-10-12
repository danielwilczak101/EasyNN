from __future__ import annotations
from typing import Generic, TypeVar
from EasyNN.model.abc import Model
from EasyNN.typing import ArrayND
import numpy as np

ArrayShape = TypeVar("ArrayShape", bound=ArrayND)


class Normalize(Model[ArrayShape, ArrayShape], Generic[ArrayShape]):
    """Normalizes the input based on the training data."""
    __lr: float
    _mean: ArrayIn = 0.0
    _variance: ArrayIn = 0.0
    _weight: float = 0.0
    _parameters: ArrayND = np.empty(0, dtype=float)
    _derivatives: ArrayND = np.empty(0, dtype=float)

    def __init__(self: Normalize[ArrayShape], lr: float = 1e-2) -> None:
        self.__lr = lr

    @property
    def mean(self: Normalize[ArrayShape]) -> ArrayShape:
        """Returns an approximation for the mean of the provided training data."""
        return self._mean / self._weight

    @property
    def variance(self: Normalize[ArrayShape]) -> ArrayShape:
        """Returns an approximation for the variance of the provided training data."""
        return self._variance / self._weight

    @property
    def deviation(self: Normalize[ArrayShape]) -> ArrayShape:
        """Returns an approximation for the deviation of the provided training data."""
        return np.sqrt(self.variance)

    def __forward__(self: Normalize[ArrayShape], x: ArrayShape) -> ArrayShape:
        if self.command.startswith("on_training_"):
            self._weight += self.__lr * (1 - self._weight)
            self._mean += self.__lr * (x.mean(axis=0) - self._mean)
            self._variance += self.__lr * (((x - self.mean) ** 2).mean(axis=0) - self._variance)
        elif self._weight == 0.0:
            return x
        return (x - self.mean) / self.deviation

    def __backward__(self: Normalize[ArrayShape], dy: ArrayShape, y: ArrayShape = None, use_y: bool = False) -> ArrayShape:
        if self._weight == 0.0:
            return dy
        return dy / self.deviation

    def __setup__(self: Normalize[ArrayShape]) -> None:
        pass
