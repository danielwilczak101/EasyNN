from __future__ import annotations
from EasyNN.model.abc import Model
from EasyNN.model.dense.bias import Bias
from EasyNN.model.dense.weight import Weight
from EasyNN.model.sequential import Sequential
from EasyNN.typing import Array1D


class DenseLayer(Model):
    """
    Create a Dense Layer with an amount of neurons.
    Dense Layers are a combination of a Weight followed
    by a Bias.
    """
    _sequential: Sequential
    neurons: int

    def __init__(self: DenseLayer, neurons: int) -> None:
        self.neurons = neurons
        self._sequential = Sequential(Weight(neurons=self.neurons), Bias(self.neurons))

    @property
    def bias(self: DenseLayer) -> Weight:
        """The underlying Bias model for a Dense Layer."""
        return self._sequential.layers[1]

    @property
    def _derivatives(self: DenseLayer) -> Array1D:
        """Dense Layer derivatives are a view into the underlying sequential model."""
        return self._sequential._derivatives

    @_derivatives.setter
    def _derivatives(self: DenseLayer, derivatives: Array1D) -> None:
        self._sequential._derivatives = derivatives

    @property
    def _parameters(self: DenseLayer) -> Array1D:
        """Dense Layer parameters are a view into the underlying sequential model."""
        return self._sequential._parameters

    @_parameters.setter
    def _parameters(self: DenseLayer, parameters: Array1D) -> None:
        self._sequential._parameters = parameters

    @property
    def weight(self: DenseLayer) -> Weight:
        """The underlying Weight model for a Dense Layer."""
        return self._sequential.layers[0]

    def __backward__(self: DenseLayer, dy: Array1D) -> Array1D:
        return self._sequential.backward(dy)

    def __forward__(self: DenseLayer, x: Array1D) -> Array1D:
        return self._sequential(x)

    def __setup__(self: DenseLayer) -> None:
        pass
