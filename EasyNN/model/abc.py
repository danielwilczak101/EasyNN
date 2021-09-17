from __future__ import annotations
from abc import ABC, abstractmethod
from functools import update_wrapper
from typing import Any, Type, Union
from EasyNN.typing import Array1D, ArrayND, ArrayLike1D, ArrayLikeND, DimensionSize, ShapeND
import numpy as np
import sys


class Model(ABC):
    """
    TODO: documentation.
    """
    _derivatives: Array1D
    _parameters: Array1D
    _setup_flag: bool = True
    _x: ArrayND
    shape_in: ShapeND = (Any, ...)
    shape_out: ShapeND = (Any, ...)

    def __init_subclass__(cls: Type[Model], **kwargs: Any) -> None:
        """
        Initiallize subclasses by setting up the documentation for dunder methods
        which don't have any documentation by using the default documentation.
        """
        for dunder in vars(cls).keys() & Model.__abstractmethods__:
            if getattr(cls, dunder).__doc__ is None:
                update_wrapper(getattr(cls, dunder), getattr(Model, dunder), updated=())
        super().__init_subclass__(**kwargs)

    @property
    def derivatives(self: Model) -> Array1D:
        """Model derivatives are a 1D array used to store parameter derivatives during model.backward(dy)."""
        return self._derivatives

    @derivatives.setter
    def derivatives(self: Model, derivatives: ArrayLike1D) -> None:
        if hasattr(self, "derivatives"):
            self._derivatives[...] = np.reshape(derivatives, -1)
        else:
            self._derivatives = np.asarray(derivatives, dtype=float).reshape(-1)

    @property
    def parameters(self: Model) -> Array1D:
        """Model parameter values are a 1D array which can be modified to change the model."""
        return self._parameters

    @parameters.setter
    def parameters(self: Model, parameters: ArrayLike1D) -> None:
        if hasattr(self, "parameters"):
            self._parameters[...] = np.reshape(parameters, -1)
        else:
            self._parameters = np.asarray(parameters, dtype=float).reshape(-1)

    @property
    def x(self: Model) -> ArrayND:
        """
        Stores the x values from the previous call for future use,
        such as backpropagation. Also sets the shape_in.
        """
        return self._x

    @x.setter
    def x(self: Model, x: ArrayLikeND) -> None:
        self._x = np.asarray(x, dtype=float)

    @x.deleter
    def x(self: Model) -> None:
        del self._x

    def __call__(self: Model, x: ArrayLikeND) -> ArrayLikeND:
        """
        Implements y = model(x) for feed-forward propagation.
        Parses the input as a numpy array before using the forward implementation method.
        """
        self.x = x
        self.setup()
        return self.__forward__(self.x)

    def backward(self: Model, dy: ArrayLikeND) -> ArrayLikeND:
        """
        Implements dx = model.backward(dy) for backpropagation.
        Parses the input as a numpy array before using the backward implementation method.
        """
        return self.__backward__(np.asarray(dy, dtype=float))

    def fit(self: Model, x: ArrayLikeND, y: ArrayLikeND) -> None:
        """
        TODO: Attempt to fit the (x, y) values provided so that model(x) is almost y.
        """

    def setup(self: Model) -> None:
        """
        During y = model(x), self.setup() is called to setup the values, derivatives,
        shapes in, and shapes out. Afterwards, if successful, self._setup_flag is set to
        False, and additional calls will do nothing.
        """
        if self._setup_flag:
            self.__setup__()
            self._setup_flag = False

    @abstractmethod
    def __backward__(self: Model, dy: ArrayND) -> ArrayLike:
        """
        Implements the backpropagation after the input has been parsed.

        By default, uses the backward-batch method.
        """
        return self.__backward_batch__(dy)

    def __backward_batch__(self: Model, dy: ArrayND) -> ArrayLike:
        """Implements the backpropagation for whole batches at a time."""
        raise NotImplementedError

    @abstractmethod
    def __forward__(self: Model, x: ArrayND) -> ArrayLike:
        """
        Implements the feed-forward propagation after the input has been parsed.

        By default, uses the forward-batch method.
        """
        return self.__forward_batch__(x)

    def __forward_batch__(self: Model, x: ArrayND) -> ArrayLike:
        """Implements the feed-forward propagation for whole batches at a time."""
        raise NotImplementedError

    @abstractmethod
    def __setup__(self: Model) -> None:
        """
        Implements the setup procedure for the values, derivatives, shapes in, and shapes
        out.
        """
        raise NotImplementedError
