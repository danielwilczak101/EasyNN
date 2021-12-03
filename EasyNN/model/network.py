"""
TODO: Documentation.
"""
from __future__ import annotations
from itertools import accumulate
import numpy as np
from typing import Callable, Generic, Iterator, Optional, Union, TypeVar, overload
from EasyNN.model.abc import Model, base_array
from EasyNN.model.dense_layer import DenseLayer
from EasyNN.typing import Array1D, ArrayND

ArrayIn = TypeVar("ArrayIn", bound=ArrayND)
ArrayOut = TypeVar("ArrayOut", bound=ArrayND)


class Network(Model[ArrayIn, ArrayOut], Generic[ArrayIn, ArrayOut]):
    """
    A full Network model, which chains together several layers into one.

    If a layer is provided as an int, then base_layer(int) is used.
    Otherwise if a layer is provided as a Model, then that layer is used.
    Otherwise layer() is used e.g. ReLU -> ReLU().

    Example:
        >>> model = Network(128, ReLU, 10, SoftMax())
        >>> # 128 -> DenseLayer(128);  for an int input.
        >>> # ReLU -> ReLU();          for a class input.
        >>> # SoftMax() -> SoftMax();  for an already initialized layer.
    """
    __parameters: Array1D
    __derivatives: Array1D
    _layers: list[Model]

    def __init__(self: Network[ArrayIn, ArrayOut], *layers: Union[int, Model, Callable[[], Model]], base_layer: Callable[[int], Model] = DenseLayer) -> None:
        self._layers = [
            self.as_layer(layer, base_layer)
            for layer in layers
        ]

    def add(self: Network, layer: Union[int, Model, Callable[[], Model]], base_layer: Callable[[int], Model] = DenseLayer) -> None:
        """Append another layer to the end of the layers."""
        self._layers.append(self.as_layer(layer, base_layer))

    @staticmethod
    def as_layer(layer: Union[int, Model, Callable[[], Model]], base_layer: Callable[[int], Model] = DenseLayer) -> Model:
        """
        Parse the layer as an actual model.

        Examples
        --------
        >>> DenseNetwork.as_layer(128)        # DenseLayer(128);  for an int input.
        >>> DenseNetwork.as_layer(ReLU)       # ReLU();           for a class input.
        >>> DenseNetwork.as_layer(SoftMax())  # SoftMax();        for an already initialized layer.
        """
        # SoftMax() -> SoftMax().
        if isinstance(layer, Model):
            return layer
        # SoftMax -> SoftMax().
        elif callable(layer):
            return layer()
        # 128 -> DenseLayer(128)
        else:
            return base_layer(layer)

    @property
    def layers(self: Network[ArrayIn, ArrayOut]) -> tuple[Model, ...]:
        """Returns the layers of the network. Includes nested layers."""
        return tuple([
            sub_layer
            for layer in self._layers
            for sub_layer in layer.layers
        ])

    @property
    def _default_loss(self: Network[ArrayIn, ArrayOut]) -> Callable[[], Loss[ArrayIn, ArrayOut]]:
        """By default, use the loss that the last layer has by default (e.g. if LogLikelihood if the last layer is SoftMax)."""
        return self.layers[-1]._default_loss

    @property
    def _parameters(self: Network[ArrayIn, ArrayOut]) -> Array1D:
        """Provides views into the layers' parameters."""
        return self.__parameters

    @_parameters.setter
    def _parameters(self: Network[ArrayIn, ArrayOut], parameters: Array1D) -> None:
        if not all(hasattr(layer, "parameters") for layer in self.layers):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute 'parameters' because a layer did not setup its parameters,\n"
                f"give the model some data to set itself up.\n"
                f"Example:\n"
                f">>> model(np.empty(28 * 28))     # Setting the input size."
                f">>> model(model.training[0][0])  # Using the training dataset."
            )
        self.__parameters = parameters
        i = 0
        for layer in self.layers:
            layer._parameters = parameters[i:i+len(layer.parameters)]
            i += len(layer.parameters)

    @property
    def _derivatives(self: Network[ArrayIn, ArrayOut]) -> Array1D:
        """Provides views into the layers' derivatives."""
        return self.__derivatives

    @_derivatives.setter
    def _derivatives(self: Network[ArrayIn, ArrayOut], derivatives: Array1D) -> None:
        self.__derivatives = derivatives
        i = 0
        for layer in self.layers:
            layer._derivatives = derivatives[i:i+len(layer.derivatives)]
            i += len(layer.derivatives)

    def get_arrays(self) -> dict[str, ArrayND]:
        """Returns the arrays stored in the model."""
        arrays = {
            f"{name}_{i}": arr
            for i, layer in enumerate(self.layers)
            for name, arr in layer.get_arrays().items()
            if name != "parameters"
        }
        if hasattr(self, "parameters"):
            arrays["parameters"] = self.parameters
        return arrays

    def set_arrays(self, *, parameters: ArrayND = None, **layer_arrays: ArrayND) -> None:
        """Sets the arrays stored in the model."""
        # Set the parameters, if any.
        if parameters is not None:
            self.parameters = parameters
        # Set the arrays for each individual layer.
        for i, layer in enumerate(self.layers):
            # Only use the arrays which match the current layer.
            layer.set_arrays(**{
                name.removesuffix(f"_{i}"): arr
                for name, arr in layer_arrays.items()
                if name.endswith(f"_{i}")
            })

    def __forward__(self: Network[ArrayIn, ArrayOut], x: ArrayIn) -> ArrayOut:
        # Pass the value through every layer.
        for layer in self._layers:
            x = layer(x)
        # After every layer is setup, setup the whole network.
        self.__setup__()
        return x

    def __backward__(self: Network[ArrayIn, ArrayOut], dy: ArrayOut, y: ArrayOut = None, use_y: bool = False):
        # Pass the derivative through every layer in reversed order.
        for layer in reversed(self._layers):
            dy = layer.backward(dy, y, use_y)
            # Only use y for the last layer, if at all.
            use_y = False
        return dy

    def __setup__(self: Network[ArrayIn, ArrayOut]) -> None:
        """Setup the Network parameter/derivative arrays as well as the arrays for each layer."""
        # Update the parameters every layer is setup,
        # but some are not setup to be the same as the DenseLayer parameters.
        if (
            all(hasattr(layer, "parameters") for layer in self.layers)
            and any(base_array(getattr(self, "parameters", None)) is not base_array(layer.parameters) for layer in self.layers)
        ):
            self._parameters = np.concatenate([layer.parameters for layer in self.layers], axis=0)
        # Update the derivatives every layer is setup,
        # but some are not setup to be the same as the DenseLayer derivatives.
        if (
            all(hasattr(layer, "derivatives") for layer in self.layers)
            and any(base_array(getattr(self, "derivatives", None)) is not base_array(layer.derivatives) is not None for layer in self.layers)
        ):
            self._derivatives = np.concatenate([layer.derivatives for layer in self.layers], axis=0)
