from typing import List, Sequence, Union
import random
import numpy as np


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
    - shape: ().  (default)
    - may be accessed using Point.points.
    - should be updated using Point.points = new_point.
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
    # - zero tensors   #
    # - random tensors #
    #==================#


    def __init__(
            self,
            shape: Sequence[int] = (),
    ):
        """Initialize 0's with the given shape. (Scalar shape = () by default.)"""

        # Initialize values directly,
        # skipping properties in case the
        # properties are recursively defined
        # (which they frequently are).
        self.shape = shape
        self._points = self.zeros(shape)
        self._derivatives = self.zeros(shape)


    def zeros(self, shape: Sequence[int]) -> TensorType:
        """Creates numpy arrays filled with 0's of the right shape."""
        return np.zeros(shape)


    def randomize(
            self,
            low: float = -10,
            high: float = 10,
            tensor: Union[None, TensorType] = None,
    ) -> TensorType:
        """Randomizes the points and derivatives."""

        if tensor is None:
            self.points = self.randomize(low, high, self.points)
            return self.points

        return np.random.uniform(low, high, np.shape(tensor))


    #========================#
    # String representations #
    #========================#


    def __str__(self) -> str:
        """Casts Point to string, showing Point.points"""
        return str(self.points)


    #=========#
    # Methods #
    #=========#


    def optimize(self):
        """
        Modifies the points using
                points -= gradient
        """
        self.points -= self.gradient


    #============#
    # Properties #
    #============#


    @property
    def points(self) -> TensorType:
        """Property allowing special points modifying."""
        return self.get_points()


    @points.setter
    def points(self, new_points: TensorType):
        """Property setter for points."""
        self.set_points(new_points)


    @property
    def derivatives(self) -> TensorType:
        """Property allowing special derivatives modifying."""
        return self.get_derivatives()


    @derivatives.setter
    def derivatives(self, new_derivatives: TensorType):
        """Property setter for derivatives."""
        return self.set_derivatives(new_derivatives)


    @property
    def gradient(self) -> TensorType:
        """Property only for getting gradient used in gradient descent."""
        return self.get_derivatives()


    def get_points(self) -> TensorType:
        """Customizable .points getter."""
        return self._points


    def set_points(self, new_points: TensorType):
        """Customizable .points setter."""
        self._points = np.array(new_points)


    def get_derivatives(self) -> TensorType:
        """Customizable .derivatives getter."""
        return self._derivatives


    def set_derivatives(self, new_derivatives: TensorType):
        """Customizable .points setter."""
        self._derivatives = np.array(new_derivatives)


class PointList(Point):
    """Subclass of Point, using mutable lists to allow mutable shapes."""


    #============#
    # Attributes #
    #============#


    shape: Sequence[int]
    """The shape of the points stored."""

    points: TensorType
    """The points stored."""

    derivatives: TensorType
    """The partial derivatives stored."""


    #==============================#
    # Initializing                 #
    # - Point          (inherited) #
    # - zero tensors   (modified)  #
    # - random tensors (inherited) #
    #==============================#


    def zeros(self, shape: Sequence[int]) -> TensorType:
        """Creates lists filled with 0's of the right shape."""
        return self.to_list(np.zeros(shape))


    #=========#
    # Methods #
    #=========#


    def optimize(self):
        """
        Modifies the points using
                points -= gradient
        """
        self.points = self.sub(self.points, self.gradient)


    #================#
    # Tensor Methods #
    #================#


    def to_list(self, tensor: TensorType) -> TensorType:
        """Converts tensor to python lists."""

        # lists of tensors
        try:
            return [self.to_list(x) for x in tensor]

        # scalars
        except TypeError:
            return float(tensor)


    def sub(self, tensor1: TensorType, tensor2: TensorType) -> TensorType:
        """Subtract two tensors component-wise."""

        # scalars
        try:
            return tensor1 - tensor2

        # lists of tensors x1 and x2
        except TypeError:
            return [self.sub(x1, x2) for x1, x2 in zip(tensor1, tensor2)]
                

    #============#
    # Properties #
    #============#


    def set_points(self, new_points: TensorType):
        """Customizable .points setter."""
        self._points = self.to_list(new_points)


    def set_derivatives(self, new_derivatives: TensorType):
        """Customizable .points setter."""
        self._derivatives = self.to_list(new_derivatives)
