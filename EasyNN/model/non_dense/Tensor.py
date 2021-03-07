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


    def __new__(cls, values: TensorLike) -> Tensor:
        """Create a new tensor object unless the input is already a tensor."""

        if isinstance(values, cls):
            return values
        else:
            return super(Tensor, cls).__new__(cls)


    def __init__(self, values: TensorLike):
        """Initialize a tensor object from values."""

        # Check if already initialized, use tensor[...] = ... to fill tensor
        try:
            self.values
            self.shape
            return
        except AttributeError:
            pass

        # check for float value
        try:
            self.values = float(values)
            self._shape = ()

        # otherwise list of values
        except TypeError:

            # convert to list of tensors
            values = [Tensor(row) for row in values]

            # get the shape of the tensors
            tensor_shapes = {tensor.shape for tensor in values}

            # there are no tensors
            if len(tensor_shapes) == 0:
                self.values = values
                self._shape = (0,)

            # 2 or more tensors have different shapes
            elif len(tensor_shapes) > 1:
                raise ValueError("Sizes do not match")

            # everything matches
            else:
                # get the tensors' shapes
                shape, = tensor_shapes

                self.values = values
                self._shape = (len(values),) + shape


    @classmethod
    def full(cls, shape: Tuple[int, ...], value: float) -> Tensor:
        """Returns a new tensor filled with the given value of the given shape."""
        return cls(np.full(shape, value))


    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> Tensor:
        """Returns a new tensor filled with zeros of the given shape."""
        return cls.full(shape, 0.0)


    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> Tensor:
        """Returns a new tensor filled with ones of the given shape."""
        return cls.full(shape, 1.0)


    @classmethod
    def random(cls, shape: Tuple[int, ...], lower: float = -10, upper: float = 10) -> Tensor:
        """Returns a new tensor filled with random values of the given shape."""
        return cls(np.random.uniform(lower, upper, shape))


    #=================#
    # Looping methods #
    #=================#


    def __len__(self) -> int:
        """Implements len(tensor)."""
        return len(self.values)


    def format_indexes(self, indexes: Sequence[Union[int, slice]]) -> Union[float, Tensor]:
        """
        Replaces an ellipsis in the indexes with slice(None, None, None)
        until len(indexes) == self.dimensions.

        Does nothing if no ellipsis is found.
        Raises an IndexError if multiple ellipses are found.
        """

        try:
            indexes = list(indexes)
        except TypeError:
            indexes = [indexes]

        ellipsis_count = indexes.count(...)

        # iterate over the indexes like normal
        if ellipsis_count == 0:
            yield from indexes
            yield from [slice(None, None, None)] * (self.dimensions - len(indexes))

        # multiple ellipses cannot be parsed
        elif ellipsis_count > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        # replace found ellipsis with slices
        else:
            for elem in indexes:
                print("ell")
                if elem == ...:
                    yield from [slice(None, None, None)] * (self.dimensions - len(indexes) + 1)
                else:
                    yield elem


    def __getitem__(self, indexes: Sequence[Union[int, slice]]) -> Union[float, Tensor]:
        """Implements tensor[i1, i2, ...]."""

        indexes = self.format_indexes(indexes)

        # extract the first index
        try:
            index = next(indexes)

        # return itself if there's no more indexes
        except StopIteration:
            return self

        # only get tensor from one row
        # when an integer index is used
        if isinstance(index, int):
            return self.values[index][indexes]

        # get tensor from multiple rows
        # when a slice index is used
        else:
            # convert index slice to range
            index_range = range(*index.indices(len(self)))

            indexes = list(indexes)
            return Tensor([self[i][indexes] for i in index_range])


    def __setitem__(t1: Tensor, indexes: Tuple[Union[int, slice], ...], t2: TensorLike):
        """Implements t1[i1, i2, ...] = t2."""

        indexes = t1.format_indexes(indexes)

        shape2 = Tensor.shape_of(t2)

        # extract the first index
        try:
            index = next(indexes)

        # if there's no more indexes
        except StopIteration:

            # update t1 with a float value
            if isinstance(t1.values, float):

                # cast t2 to float
                try:
                    t1.values = float(t2)

                # if it's not a float, figure out error
                except TypeError:

                    try:
                        iter(t2)

                        # tried to update float with list
                        raise IndexError("Wrong input shape compared to indexes or slice")

                    # tried to update float with non-list
                    except TypeError:
                        raise ValueError("Tensor only accepts float values.")

            # wrong shape
            else:
                raise IndexError("Wrong input shape compared to indexes or slice")

        # there's more indexes
        else:

            # assign to current index
            if isinstance(index, int):
                t1.values[index][indexes] = t2

            # assign to slice
            else:
                # convert index slice to range
                index_range = range(*index.indices(len(t1)))

                # check matching sizes
                try:
                    if len(index_range) != len(t2):
                        raise TypeError
                except TypeError:
                    raise IndexError("Wrong input shape compared to indexes or slice")

                # update at indexes
                indexes = list(indexes)
                for i, row in zip(index_range, t2):
                    t1.values[i][indexes] = row


    def __iter__(self) -> Iterable[Tensor]:
        """Implements looping over tensors."""
        return iter(self.values)


    def deep_iter(self) -> Iterable[float]:
        """Returns all floats in the tensor."""

        if self.is_scalar():
            yield float(self)

        else:
            for row in self:
                yield from row.deep_iter()


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


    def __truediv__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t = t1 / t2."""
        pass


    def __abs__(self) -> float:
        """Implements componentwise abs(tensor)."""
        pass


    def __neg__(self) -> Tensor:
        """Implements t1 = -t2."""
        pass


    def __eq__(t1: Tensor, t2: TensorLike) -> Tensor:
        """Implements t1 == t2."""
        pass


    def norm(self, order: float = 2) -> float:
        """Implements magnitude = tensor.norm(order)."""
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


    def __itruediv__(t1: Tensor, t2: TensorLike) -> Tensor:
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
    def shape(self):
         """Getter for tensor.shape."""
         return self._shape


    @property
    def dimensions(self):
        """Getter for tensor.dimensions based on the shape."""
        return len(self.shape)


    def is_scalar(self) -> bool:
        """Returns if the Tensor is a scalar value."""
        return self.dimensions == 0


    def is_vector(self) -> bool:
        """Returns if the Tensor is a vector value."""
        return self.dimensions == 1


    def is_matrix(self) -> bool:
        """Returns if the Tensor is a matrix value."""
        return self.dimensions == 2


    def __float__(self) -> float:
        """Implements float(tensor)."""
        return float(self.values)


    def copy(self) -> Tensor:
        """Copy the entire tensor."""
        return Tensor(self, self.shape)


    def __repr__(self) -> str:
        """Return a code string representation of the tensor."""

        def pad_lines(s: str, spaces=1) -> str:
            """Puts an extra space before every line."""
            return '\n'.join(' '*spaces + line for line in s.split('\n'))

        spaces = len("Tensor(")

        return "Tensor(" + pad_lines(str(self), spaces)[spaces:] + ")"


    def __str__(self) -> str:
        """Return a readable representation of the tensor."""

        def pad_lines(s: str, spaces=1) -> str:
            """Puts an extra space before every line."""
            return '\n'.join(' '*spaces + line for line in s.split('\n'))

        # for multiple dimensions,
        # indent all lines with 1 space,
        # except the first line which gets a '[',
        # and finally a ']' is applied to the end

        # if there are more than 2 dimensions,
        # 2 new lines are used between rows
        if self.dimensions > 2:
            return '[' + ",\n\n".join(pad_lines(str(row)) for row in self)[1:] + ']'

        # For matrices, only 1 new line is used between rows
        elif self.is_matrix():
            return '[' + ",\n".join(pad_lines(str(row)) for row in self)[1:] + ']'

        # array is printed as usual
        elif self.is_vector():
            return '[' + ", ".join(str(row) for row in self) + ']'

        # scalar is printed as usual
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
        try:
            float(self)
        except Exception:
            pass
        else:
            return ()

        # get the unique shapes of each row
        unique_shapes = {Tensor.shape_of(row) for row in self}

        # 2 or more rows have different shapes
        if len(unique_shapes) > 1:
            raise ValueError("Sizes do not match")

        # prepend the length of itself to the shape of each row
        else:
            # get the unique shape
            shape, = unique_shapes
            return (len(self),) + shape
