"""
TODO: Documentation.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, overload
from EasyNN.model.abc import Model, base_array
from EasyNN.model.bias import Bias
from EasyNN.model.weight import Weight
from EasyNN.typing import Array1D


class DenseLayer(Model):
    """
    Create a Dense Layer with an amount of neurons.
    Dense Layers are a combination of a Weight followed by a Bias.
    """
    __parameters: Array1D
    __derivatives: Array1D
    neurons: int
    weight: Weight
    bias: Bias

    def __init__(self: DenseLayer, neurons: int) -> None:
        if not isinstance(neurons, int):
            raise TypeError(f"expected int for neurons, got {type(neurons).__name__} instead")
        elif neurons <= 0:
            raise ValueError(f"neurons must be a positive integer, not {neurons}")
        self.neurons = neurons
        self.weight = Weight(neurons)
        self.bias = Bias(neurons)

    @property
    def last_layer(self: DenseLayer) -> Model:
        """Returns the last layer of the model."""
        return self.layers[-1].last_layer

    @property
    def _parameters(self: DenseLayer) -> Array1D:
        """Provides views into the weight/bias parameters."""
        return self.__parameters

    @_parameters.setter
    def _parameters(self: DenseLayer, parameters: Array1D) -> None:
        self.__parameters = parameters
        self.weight._parameters = parameters[:-self.neurons]
        self.bias._parameters = parameters[-self.neurons:]

    @property
    def _derivatives(self: DenseLayer) -> Array1D:
        """Provides views into the weight/bias derivatives."""
        return self.__derivatives

    @_derivatives.setter
    def _derivatives(self: DenseLayer, derivatives: Array1D) -> None:
        self.__derivatives = derivatives
        self.weight._derivatives = derivatives[:-self.neurons]
        self.bias._derivatives = derivatives[-self.neurons:]

    @property
    def layers(self: DenseLayer) -> tuple[Weight, Bias]:
        """Returns the (weight, bias) layers as a tuple."""
        return (self.weight, self.bias)

    @overload
    def __backward__(self: DenseLayer, dy: Array1D, y: Optional[Array1D] = ..., use_y: bool = ...) -> Array1D:
        ...

    @overload
    def __backward__(self: DenseLayer, dy: Array2D, y: Optional[Array2D] = ..., use_y: bool = ...) -> Array2D:
        ...

    def __backward__(self, dy, y=None, use_y=False):
        return self.weight.backward(self.bias.backward(dy))

    @overload
    def __forward__(self: DenseLayer, x: Array1D) -> Array1D:
        ...

    @overload
    def __forward__(self: DenseLayer, x: Array2D) -> Array2D:
        ...

    def __forward__(self, x):
        result = self.bias(self.weight(x))
        self.__setup__()
        return result

    def __setup__(self: DenseLayer) -> None:
        # Update the parameters every layer is setup,
        # but some are not setup to be the same as the DenseLayer parameters.
        if (
            all(hasattr(layer, "parameters") for layer in self.layers)
            and any(base_array(getattr(self, "parameters", None)) is not base_array(layer.parameters) for layer in self.layers)
        ):
            self._parameters = np.append(self.weight.parameters, self.bias.parameters)
        # Update the derivatives every layer is setup,
        # but some are not setup to be the same as the DenseLayer derivatives.
        if (
            all(hasattr(layer, "derivatives") for layer in self.layers)
            and any(base_array(getattr(self, "derivatives", None)) is not base_array(layer.derivatives) is not None for layer in self.layers)
        ):
            self._derivatives = np.append(self.weight.derivatives, self.bias.derivatives)
