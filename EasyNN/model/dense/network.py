from __future__ import annotations
from typing import Callable, Union
from EasyNN.model.abc import Model
from EasyNN.model.dense.layer import DenseLayer
from EasyNN.model.sequential import Sequential


class DenseNetwork(Sequential):
    """
    Allows integers to be used for layers to shortcut to a DenseLayer
    with that many neurons in that layer.

    Example
    -------
    Create a dense network with 100 neurons and a ReLU activation,
    followed by 50 neurons and another ReLU activation, and finally
    10 neurons and a SoftMax activation.
    >>> model = DenseNetwork(100, ReLU, 50, ReLU, 10, SoftMax)
    """

    def __init__(self: DenseNetwork, *layers: Union[int, Model, Callable[[], Model]]) -> None:
        super().__init__(*[
            DenseLayer(model) if isinstance(model, int) else model
            for model in layers
        ])

    def add(self: DenseNetwork, model: Union[int, Model, Callable[[], Model]]) -> None:
        """Append another layer to the end of the layers."""
        super().add(DenseLayer(model) if isinstance(model, int) else model)
