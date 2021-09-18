"""
Contains all of the type-hint related things.
"""
from nptyping import NDArray
from numpy.typing import ArrayLike
from typing import Any, Sequence, Type, Union

ellipsis = type(...)
DimensionSize = Union[int, Type[Any]]  # Should be Literal[Any].
Array1D = NDArray[(Any,), float]
Array2D = NDArray[(Any, Any), float]
Array3D = NDArray[(Any, Any, Any), float]
ArrayND = NDArray[(Any, ...), float]
ArrayLike1D = Union[Array1D, Sequence[float]]
ArrayLike2D = Union[Array2D, Sequence[Sequence[float]]]
ArrayLike3D = Union[Array3D, Sequence[Sequence[Sequence[float]]]]
ArrayLikeND = ArrayLike
Shape1D = tuple[DimensionSize]
Shape2D = tuple[DimensionSize, DimensionSize]
Shape3D = tuple[DimensionSize, DimensionSize, DimensionSize]
ShapeND = tuple[Union[DimensionSize, ellipsis], ...]
Sample = tuple[ArrayND, ...]
