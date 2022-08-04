"""
Contains all of the type-hint related things.
"""
from nptyping import Float, NDArray, Shape
from numpy.typing import ArrayLike
from typing import Any, Callable, Literal, Sequence, Type, TypeVar, Union

T = TypeVar("T")

Factory = Union[T, Callable[[], T]]
Array1D = NDArray[Shape["*"], Float]
Array2D = NDArray[Shape["*, *"], Float]
Array3D = NDArray[Shape["*, *, *"], Float]
ArrayND = NDArray[Any, Float]
ArrayLike1D = Union[Array1D, Sequence[float]]
ArrayLike2D = Union[Array2D, Sequence[Sequence[float]]]
ArrayLike3D = Union[Array3D, Sequence[Sequence[Sequence[float]]]]
ArrayLikeND = ArrayLike
Sample = tuple[ArrayND, ...]

Callback = Callable[[], Any]
Command = Literal[
    "off",
    "on_optimization_start",
    "on_optimization_end",
    "on_training_start",
    "on_training_end",
    "on_testing_start",
    "on_testing_end",
    "on_validation_start",
    "on_validation_end",
    "on_epoch_start",
    "on_epoch_end",
]
