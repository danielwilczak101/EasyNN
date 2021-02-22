from __future__ import annotations
from typing import Iterable, Sequence, List, Tuple
import random
import numpy as np


TensorLike = Union[float, np.ndarray, Sequence[TensorLike]]


class Tensor:
    """Tensor class using python lists for mutable shapes."""

    values: Union[float, List[Tensor]]
    """The values stored in the tensor."""

    shape: Sequence[int]
    """The shape of the tensor."""


    #======================#
    # Initializing tensors #
    #======================#


    def __init__(self, values: TensorLike, shape: Sequence[int] = None):
        """Initialize a tensor object from values."""
        pass


    @staticmethod
    def zeros(shape: Sequence[int]) -> Tensor:
        """Returns a new tensor filled with zeros of the given shape."""
        pass


    @staticmethod
    def random(shape: Sequence[int], lower: float = 0, upper: float = 1) -> Tensor:
        """Returns a new tensor filled with random values of the given shape."""
        pass


    #=================#
    # Looping methods #
    #=================#


    def __len__(self) -> int:
        """Implements len(tensor)."""
        pass


    def __getitem__(self, *indexes: Tuple[Union[int, slice], ...]) -> Union[float, Tensor]:
        """Implements tensor[i1, i2, ...]."""
        pass


    def __setitem__(t1: Tensor, *indexes: Tuple[Union[int, slice], ...], t2: TensorLike):
        """Implements t1[i1, i2, ...] = t2."""
        pass



    def __delitem__(t1: Tensor, *indexes: Tuple[Union[int, slice], ...]):
        """Implements del t1[:, :, ..., i]."""
        pass


    def __iter__(self) -> Iterable[Tensor]:
        """Implements looping over tensors."""
        pass


    #===================#
    # Tensor operations #
    #===================#


    # Note: if t2 is a float, just apply it to everything in t1
    # e.g. Tensor([1, 2, 3]) + 1 == Tensor([2, 3, 4])


    def __add__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t = t1 + t2."""
        pass


    def __sub__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t = t1 - t2."""
        pass


    def __mul__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t = t1 * t2."""
        pass


    def __div__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t = t1 / t2."""
        pass


    def __abs__(self) -> float:
        """Implements magnitude = abs(tensor)."""
        pass


    def norm(self, order: float = 2) -> float:
        """Implements magnitude = tensor.norm(order)."""
        pass


    def __neg__(self) -> Tensor:
        """Implements t1 = -t2."""
        pass


    def __eq__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t1 == t2."""
        pass


    def matmul(t1: Tensor, t2: TensorLike) -> Tensor:
        """Matrix multiplies t1 and t2, where tensor.dimensions <= 2."""
        pass


    #============================#
    # Tensor in-place operations #
    #============================#


    def __iadd__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t1 += t2."""
        pass
        return t1


    def __isub__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t1 -= t2."""
        pass
        return t1


    def __imul__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t1 *= t2."""
        pass
        return t1


    def __idiv__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t1 /= t2."""
        pass
        return t1


    def normalize(self, order: float = 2):
        """Implements tensor.normalize(order)"""
        pass


    #===============#
    # Miscellaneous #
    #===============#


    @property
    def dimensions(self):
        """Implements tensor.dimensions based on the shape."""
        pass


    def copy(self) -> Tensor:
        """Copy the entire tensor."""
        pass
