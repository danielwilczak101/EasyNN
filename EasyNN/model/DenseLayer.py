from EasyNN.model.Stack import Stack
from EasyNN.model.Weights import Weights
from EasyNN.model.Biases import Biases
from EasyNN.model.activation.Activation import Activation


class DenseLayer(Stack):
    """Dense Layer Model, which is a Stack of weights, bias, and activation functions."""

    def __init__(self, inputs: int, outputs: int, activation: Activation) -> None:
        """
        Create a Dense Layer, provided the amount of inputs and outputs.

        Parameters
        ----------
        inputs : int
            The amount of inputs into the dense layer.
        outputs : int
            The amount of outputs from the dense layer.
        activation : Activation
            The activation layer used.

        Notes
        -----
        Creates a Stack using weights, biases, and activations.

        Example
        -------
        Create a Dense Layer taking 100 inputs, producing 40 outputs, with ReLU.
        >>> model = DenseLayer(100, 40, ReLU)
        """
        super().__init__([Weights(inputs, outputs), Biases(outputs), activation])
