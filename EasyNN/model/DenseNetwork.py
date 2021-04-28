from typing import Iterable, Tuple, Union, Callable
from EasyNN.model.Stack import Stack
from EasyNN.model.DenseLayer import DenseLayer
from EasyNN.model.activation.Activation import Activation
from EasyNN.model.activation.ReLU import ReLU


class DenseNetwork(Stack):
    """Dense Network Model, which is a Stack of Dense Layers."""

    def __init__(
        self,
        layers: Iterable[Union[int, Tuple[int, Activation]]],
        default_activation: Callable[[], Activation] = lambda: ReLU(),
    ) -> None:
        """
        Create a Dense Network, provided the layers.

        Parameters
        ----------
        layers : Iterable[Union[int, Tuple[int, Activation]]]
            The layer specifications, in the format of
                [network_inputs, ..., amount_of_nodes_in_layer, ..., network_outputs]
            where amount_of_nodes_in_layer and network_outputs can be (int, Activation)
            to specify the layer's activation.
        outputs : int
            The amount of outputs from the dense layer.
        default_activation : Callable[[], Activation] = lambda: ReLU
            The default activation used for layers that don't have activations specified.

        Notes
        -----
        Creates a Stack using weights, biases, and activations.

        Example
        -------
        The following creates a model with 100 inputs, two hidden layers with 40 nodes,
        and 10 outputs. The second hidden layer uses the ReLU activation.
        >>> model = DenseNetwork([100, 40, (40, ReLU), 10])
        """

        inputs = list(layers)
        for i, layer in enumerate(inputs):
            if isinstance(layer, int):
                inputs[i] = (layer, default_activation())
        outputs = inputs[1:]

        super().__init__([
            DenseLayer(layer_in[0], layer_out[0], layer_out[1])
            for layer_in, layer_out
            in zip(inputs, outputs)
        ])
