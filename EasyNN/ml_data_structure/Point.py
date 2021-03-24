"""
Points store values and derivatives in a singular tensor.
Features Point.values and Point.derivatives.

>>> p = Point(np.array([[1, 2, 3],
...                     [4, 5, 6]]))
>>> print(p.values)
[1, 2, 3]
>>> print(p.derivatives)
[4, 5, 6]
>>> p = Point(Tensor([[1, 2, 3],
...                   [4, 5, 6]]))
>>> print(p.values)
[1, 2, 3]
>>> print(p.derivatives)
[4, 5, 6]
"""

from typing import List, Sequence, Union
import random
import numpy as np
from EasyNN.ml_data_structure.Tensor import TensorLike, Tensor


class Point:
    """
    Stores values and derivatives in a single Tensor while providing
    @property usage for getting values and derivatives in a readable format.
    """

    items: TensorLike
    """Stores the values as items[0] and derivatives as items[1]."""


    #====================================================#
    # Dunder methods to make self behave like self.items #
    #====================================================#


    def __getattr__(self, name):
        """Defaults self.attribute to self.items.attribute."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.items, name)


    def __setattr__(self, name, value):
        """Defaults self.attribute to self.items.attribute."""
        try:
            return super().__setattr__(name, value)
        except AttributeError:
            return setattr(self.items, name, value)


    def __hasattr__(self, name):
        """Defaults checking self.attribute to self.items.attribute."""
        return super().__hasattr__(name) or hasattr(self.items, name)


    def __init__(
            self,
            items: TensorLike,
    ):
        """Store the items."""
        self.items = items


    def __iter__(self):
        """Implements iter(self)."""
        return iter(self.items)


    def __len__(self):
        """Implements len(self)."""
        return len(self.items)


    def __getitem__(self, index):
        """Implements self[index]."""
        return self.items[index]


    def __setitem__(self, index, value):
        """Implements self[index] = value."""
        self.items[index] = value


    def __str__(self) -> str:
        """Implements str(self)."""
        return str(self.items)


    def __repr__(self) -> str:
        """Implements repr(self)."""
        return repr(self.items)


    #====================================================#
    # Additional properties for .values and .derivatives #
    #====================================================#


    @property
    def values(self) -> TensorLike:
        """Property for getting values from items."""
        return self.items[0]


    @values.setter
    def values(self, new_values: TensorLike):
        """Set values in items."""
        self.items[0] = new_values


    @property
    def derivatives(self) -> TensorLike:
        """Property for getting derivatives from items."""
        return self.items[1]


    @derivatives.setter
    def derivatives(self, new_derivatives: TensorLike):
        """Set derivatives in items."""
        self.items[0] = new_derivatives
