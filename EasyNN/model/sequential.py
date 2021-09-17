from __future__ import annotations
from itertools import accumulate
import numpy as np
from typing import Callable, Iterator, Union
from EasyNN.model.abc import Model
from EasyNN.typing import Array1D, ArrayND


class Sequential(Model):
    """
    A Sequential Model, which chains together several layers into one.

    If a layer is provided as a Model, then layer is used.
    Otherwise layer() is used e.g. ReLU -> ReLU().
    """
    layers: list[Model]

    def __init__(self: Sequential, *layers: Union[Model, Callable[[], Model]]) -> None:
        self.layers = [
            layer if isinstance(layer, Model) else layer()
            for layer in layers
        ]
        # Setup base arrays.
        self._derivatives = np.empty(0)
        self._parameters = np.empty(0)

    def add(self: Sequential, layer: Union[Model, Callable[[], Model]]) -> None:
        """Append another layer to the end of the layers."""
        self.layers.append(layer if isinstance(layer, Model) else layer())

    def __forward__(self: Sequential, x: ArrayND) -> ArrayND:
        # Pass the value through every layer.
        for layer in self.layers:
            x = layer(x)
        # Setup the layers again after the forward if some models changed (due to getting setup later).
        if not all(params.base is self.parameters for params in self.layer_parameters()):
            self.__setup__()
        return x

    def __backward__(self: Sequential, dy: ArrayND) -> ArrayND:
        # Pass the derivative through every layer in reversed order.
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def layer_derivatives(self: Sequential) -> Iterator[Array1D]:
        """
        Generates the derivatives from each layer,
        or an empty array with length given by the parameters,
        or an empty 0-length array if it doesn't have any.
        """
        for layer in self.layers:
            yield getattr(layer, "derivatives", np.empty(len(getattr(layer, "parameters", []))))

    def layer_parameters(self: Sequential) -> Iterator[Array1D]:
        """
        Generates the parameters from each layer,
        or an empty array with length given by the derivatives,
        or an empty 0-length array if it doesn't have any.
        """
        for layer in self.layers:
            yield getattr(layer, "parameters", np.empty(len(getattr(layer, "derivatives", []))))

    def __setup__(self: Sequential) -> None:
        """Setup the Sequential parameter/derivative arrays as well as the arrays for each layer."""
        # Don't setup until every layer has its arrays setup.
        if not all(hasattr(layer, "derivatives") and hasattr(layer, "parameters") for layer in self.layers):
            return
        # Setup the sequential parameters/derivatives.
        total_params = sum(len(params) for params in self.layer_parameters())
        self._derivatives = np.empty(total_params, dtype=float)
        self._parameters = np.empty(total_params, dtype=float)
        # Setup the arrays for each layer.
        for i, derivs, params, layer in zip(
            accumulate((len(params) for params in self.layer_parameters()), initial=0),
            self.layer_derivatives(),
            self.layer_parameters(),
            self.layers,
        ):
            if hasattr(layer, "parameters"):
                # Get the data from each array.
                self.derivatives[i:i+len(params)] = derivs
                self.parameters[i:i+len(params)] = params
                # Replace the model arrays with views into the sequential arrays.
                layer._derivatives = self.derivatives[i:i+len(params)]
                layer._parameters = self.parameters[i:i+len(params)]
