from typing import List, Sequence, Union
import numpy as np
import random


TensorType = Union[np.ndarray, float, List['TensorType']]
"""
Tensor Type is either a float or list of Tensors
(a multidimensional array, like numpy arrays).
"""


class Point:
    """
    Base class for handling points.
    Additionally stores derivatives and other parameters
    and various methods which may be useful.
    Uses numpy arrays.

    Note on scalar values:
    - shape: (1,).
    - may be accessed using Point.points[0].
    - should be updated using Point.points = [new_point].
    """


    #============#
    # Attributes #
    #============#


    shape: Sequence[int]
    """The shape of the points stored."""

    points: np.ndarray
    """The points stored."""

    derivatives: np.ndarray
    """The partial derivatives stored."""


    #==================#
    # Initializing     #
    # - Point          #
    # - random tensors #
    #==================#


    def __init__(
            self,
            shape: Sequence[int],
    ):
        """Initialize 0's with the given shape."""

        # Initialize values directly,
        # skipping properties in case the
        # properties are recursively defined
        # (which they frequently are).
        self.shape = shape
        self._points = np.zeros(shape)
        self._derivatives = np.zeros(shape)


    def randomize(
            self,
            low: float = -10,
            high: float = 10,
            tensor: Union[None, np.ndarray] = None,
    ):
        """Randomizes the points and derivatives."""

        if tensor is None:
            self._points = np.random.uniform(low, high, self.shape)
            self._derivatives = np.random.uniform(low, high, self.shape)

        else:
            tensor[:] = np.random.uniform(low, high, np.shape(tensor))


    #========================#
    # String representations #
    #========================#


    def __str__(self):
        """Casts Point to string, showing Point.points"""
        return str(self.points)


    #=========#
    # Methods #
    #=========#


    def optimize(self, normalizer: float = 1):
        """
        Modifies the points using
                points -= normalizer * derivatives
        """
        self.points -= normalizer * self.derivatives


    #============#
    # Properties #
    #============#


    @property
    def points(self) -> Sequence[float]:
        """Property allowing special points modifying."""
        return self._points


    @points.setter
    def points(self, new_points: Sequence[float]):
        """Property setter for points."""
        self._points = new_points


    @property
    def derivatives(self) -> Sequence[float]:
        """Property allowing special derivatives modifying."""
        return self._derivatives


    @derivatives.setter
    def derivatives(self, new_derivatives: Sequence[float]):
        """Property setter for derivatives."""
        self._derivatives = new_derivatives


class PointList(Point):
    """Subclass of Point, using mutable lists to allow mutable shapes."""


    #==================#
    # Initializing     #
    # - Point          #
    # - zero tensors   #
    # - random tensors #
    #==================#


    def __init__(
            self,
            shape: Sequence[int],
    ):
        """Initialize 0's with the given shape."""

        # Initialize values directly,
        # skipping properties in case the
        # properties are recursively defined
        # (which they frequently are).
        self.shape = shape
        self._points = self.zeros(shape)
        self._derivatives = self.zeros(shape)


    def zeros(self, shape: Sequence[int]) -> TensorType:
        """Creates lists filled with 0's of the right shape."""

        # If shape = [], return scalar 0
        if len(shape) == 0:
            return 0

        # If shape = [dimension, ...],
        # return vector of the dimension
        # filled with zeros(...)
        dimension, *shape = shape
        return [self.zeros(shape) for _ in range(dimension)]


    def randomize(
            self,
            low: float = -10,
            high: float = 10,
            tensor: Union[None, TensorType] = None,
    ) -> Union[None, TensorType]:
        """Randomizes the points and derivatives."""

        if tensor is None:
            self.randomize(low, high, self.points)
            self.randomize(low, high, self.derivatives)
            return None

        elif type(tensor) in (int, float):
            return random.uniform(low, high)

        else:
            tensor[:] = [
                self.randomize(low, high, subtensor)
                for subtensor
                in tensor
            ]
            return tensor


    #=========#
    # Methods #
    #=========#


    def optimize(self, normalizer: float = 1):
        """
        Modifies the points using
                points -= normalizer * derivatives
        """
        self.points = [
            point - normalizer * dx
            for point, dx
            in zip(self.points, self.derivatives)
        ]
