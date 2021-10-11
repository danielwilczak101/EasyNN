from __future__ import annotations
from EasyNN.typing import ArrayND
from typing import Generic, TypeVar

Array = TypeVar("Array", bound=ArrayND[float])


class Momentum(Generic[Array]):
    """Basic implementation of an exponentially moving average used for momentum."""
    _momentum: Array = 0.0
    _weight: float = 0.0
    lr: float

    @property
    def value(self: Momentum[Array]) -> Array:
        return self._momentum / self._weight

    @value.setter
    def value(self: Momentum[Array], value: Array) -> None:
        self._weight += self.lr * (1 - self._weight)
        self._momentum += self.lr * (value - self._momentum)

    def __init__(self: Momentum[Array], lr: float = 1e-2) -> None:
        self.lr = lr

    def update(self: Momentum[Array], value: Array) -> Array:
        """Update and return the momentum value."""
        self.momentum = value
        return self.momentum
