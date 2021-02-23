from __future__ import annotations
from typing import Iterable, Sequence, List, Tuple, Union
import random
import numpy as np

TensorLike = Union[float, np.ndarray, Sequence[Union['Tensor', 'TensorLike']]]


class Tensor:
    """Tensor class using python lists for mutable shapes."""

    values: Union[float, List[Tensor]]
    """The values stored in the tensor."""

    shape: Tuple[int, ...]
    """The shape of the tensor."""


    #======================#
    # Initializing tensors #
    #======================#


    def __init__(self, values: TensorLike, shape: Union[None, Tuple[int, ...]] = None):
        """Initialize a tensor object from values."""

        # initially check if the values are
        # shaped like a tensor and get the shape.
        if shape is None:
            shape = Tensor.shape_of(values)

        self.shape = shape

        # a 0-dimensional tensor is a float
        if self.dimensions == 0:
            self.values = float(values)

        # otherwise fill in recursively with lists of tensors
        else:
            shape = shape[1:]
            self.values = [Tensor(row, shape) for row in values]


    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> Tensor:
        """Returns a new tensor filled with zeros of the given shape."""
        pass


    @staticmethod
    def random(shape: Tuple[int, ...], lower: float = 0, upper: float = 1) -> Tensor:
        """Returns a new tensor filled with random values of the given shape."""
        pass


    #=================#
    # Looping methods #
    #=================#


    def __len__(self) -> int:
        """Implements len(tensor)."""
        return self.shape[0]


    def __getitem__(self, indexes: Sequence[Union[int, slice]]) -> Union[float, Tensor]:
        """Implements tensor[i1, i2, ...]."""

        if type(indexes) not in (list, tuple):
            indexes = [indexes]

        # return a copy of itself if there's no more indexes
        if len(indexes) == 0:
            return self.copy()

        # extract the first index
        index, *indexes = indexes

        # only get tensor from one row
        # when an integer index is used
        if isinstance(index, int):
            return self.values[index].__getitem__(indexes)

        # get tensor from multiple rows
        # when a slice index is used
        return Tensor([self[i].__getitem__(indexes) for i in range(*index.indices(len(self)))])


    def __setitem__(t1: Tensor, indexes: Tuple[Union[int, slice], ...], t2: TensorLike):
        """Implements t1[i1, i2, ...] = t2."""

        if type(indexes) not in (list, tuple):
            indexes = [indexes]

        shape2 = Tensor.shape_of(t2)

        # Update t1 with a float value
        if len(indexes) == 0 and isinstance(t1.values, float):
            try:
                t1.values = float(t2)
            except TypeError:
                raise ValueError("Wrong input shape compared to indexes or slice")

        # Update t1 with tensors
        elif len(indexes) == 0:
            if t1.shape != shape2:
                raise ValueError("Wrong input shape compared to indexes or slice")

            t1.values = Tensor(t2, shape2).values

        else:
            # extract the first index
            index, *indexes = indexes

            # Assign to current index
            if isinstance(index, int):
                t1.values[index].__setitem__(indexes, t2)

            # Assign to slice
            else:
                index = range(*index.indices(len(t1)))
                if len(index) != len(t2):
                    raise ValueError("Wrong input shape compared to indexes or slice")
                for i, row in zip(index, t2):
                    t1.values[i].__setitem__(indexes, row)


    def __iter__(self) -> Iterable[Tensor]:
        """Implements looping over tensors."""
        return iter(self.values)


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
        self /= self.norm(order)


    #===============#
    # Miscellaneous #
    #===============#


    @property
    def dimensions(self):
        """Implements tensor.dimensions based on the shape."""
        return len(self.shape)


    def __float__(self) -> float:
        """Implements float(tensor)."""
        return float(self.values)


    def copy(self) -> Tensor:
        """Copy the entire tensor."""
        return Tensor(self, self.shape)


    def __repr__(self) -> str:
        """Return a code string representation of the tensor."""
        return f"Tensor({self.values}, {self.shape})"


    def __str__(self) -> str:
        """Return a readable representation of the tensor."""

        # multiple dimensions        
        if self.dimensions > 1:
            return (
                "[\n"
                + ",\n".join(
                             '\n'.join(
                                       "    " + line
                                       for line
                                       in str(row).split('\n')
                             )
                             for row
                             in self
                )
                + "\n]"
            )

        # array
        elif self.dimensions == 1:
            return "[" + ", ".join(str(row) for row in self) + "]"

        # scalar
        else:
            return str(self.values)


    def shape_of(self: TensorLike) -> Tuple[int, ...]:
        """Returns the shape of a tensor-like object."""

        # Check for already saved shape
        try:
            return self.shape
        except AttributeError:
            pass

        # a float is a 0 dimensional tensor
        if type(self) in (int, float):
            return ()

        # get the unique shapes of each row
        shape = {Tensor.shape_of(row) for row in self}

        # there are no rows
        if len(shape) == 0:
            raise ValueError("Size of the last dimension can't be 0")

        # 2 or more rows have different shapes
        elif len(shape) > 1:
            raise ValueError("Sizes do not match")

        # prepend the length of itself to the shape of each row
        else:
            return (len(self),) + list(shape)[0]
