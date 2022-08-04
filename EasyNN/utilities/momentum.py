from __future__ import annotations
from copy import copy
from EasyNN.typing import ArrayND
from typing import Generic, TypeVar

Array = TypeVar("Array", bound=ArrayND)


class Momentum(Generic[Array]):
    """Basic implementation of an exponentially moving average used for momentum."""
    _weight: float = 0.0
    _value: Array = 0.0
    lr: float

    @property
    def value(self: Momentum[Array]) -> Array:
        return self._value / self._weight

    @value.setter
    def value(self: Momentum[Array], value: Array) -> None:
        self._weight += self.lr * (1 - self._weight)
        self._value += self.lr * (value - self._value)

    def __init__(self: Momentum[Array], lr: float = 1e-2) -> None:
        self.lr = lr

    def update(self: Momentum[Array], value: Array) -> Array:
        """Update and return the momentum value."""
        self.value = value
        return self.value

    def peek(self: Momentum[Array], value: Array) -> Array:
        """Peek at the result of `momentum.update(value)` without updating the momentum."""
        _weight = copy(self._weight)
        _value = copy(self._value)
        value = self.update(value)
        self._weight = _weight
        self._value = _value
        return value
