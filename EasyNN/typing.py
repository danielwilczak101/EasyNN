"""
Contains all of the type-hint related things.
"""
from nptyping import NDArray
from numpy.typing import ArrayLike
from typing import Any, Callable, Literal, Sequence, Type, TypeVar, Union

T = TypeVar("T")

Factory = Union[T, Callable[[], T]]
Array1D = NDArray[(Any,), Any]
Array2D = NDArray[(Any, Any), Any]
Array3D = NDArray[(Any, Any, Any), Any]
ArrayND = NDArray[(Any, ...), float]
ArrayLike1D = Union[Array1D, Sequence[Any]]
ArrayLike2D = Union[Array2D, Sequence[Sequence[Any]]]
ArrayLike3D = Union[Array3D, Sequence[Sequence[Sequence[Any]]]]
ArrayLikeND = ArrayLike
Sample = tuple[ArrayND, ...]

Callback = Callable[[], Any]
Command = Literal[
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
