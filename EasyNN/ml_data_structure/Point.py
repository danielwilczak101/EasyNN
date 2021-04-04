"""
Points store values and derivatives in a singular tensor.
Features Point.values and Point.derivatives.

>>> p = Point([[1, 2, 3],
...            [4, 5, 6]])
>>> print(p.values)
Point([1, 2, 3])
>>> print(p.derivatives)
Point([4, 5, 6])
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class Point(np.ndarray):
    """
    Stores values and derivatives in a single Tensor while providing
    @property usage for getting values and derivatives in a readable format.
    """

    def __new__(cls, *args, **kwargs) -> Point:
        """Create a new numpy array viewed as a Point"""
        return np.array(*args, **kwargs).view(Point)

    @property
    def values(self) -> np.ndarray:
        """Property for getting values from items."""
        return self[0].view(np.ndarray)

    @values.setter
    def values(self, new_values: ArrayLike) -> None:
        """Set values in items."""
        self[0] = new_values

    @property
    def derivatives(self) -> np.ndarray:
        """Property for getting derivatives from items."""
        return self[1].view(np.ndarray)

    @derivatives.setter
    def derivatives(self, new_derivatives: ArrayLike) -> None:
        """Set derivatives in items."""
        self[1] = new_derivatives
