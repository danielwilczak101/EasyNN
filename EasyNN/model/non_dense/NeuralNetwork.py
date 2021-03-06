"""
Generic Neural Network module. Use NeuralNetwork() to create a neural network.

>>> model = NN({
...     0: [],
...     1: [],
...     2: [0, 1],
...     3: [0, 2],
...     4: [1, 2],
... })
>>> model.nodes[2].previous_weights[0] = [1, 1]
>>> model.nodes[3].previous_weights[0] = [1, 1]
>>> model.nodes[4].previous_weights[0] = [1, 1]
>>> print(model([1, 1]))
[3.0, 3.0]
"""

# Special import which allows type hints
# involving the class itself to be inside.
from __future__ import annotations

# Type hint structures
from typing import Dict, Tuple, List, Sequence, Callable

import random
import numpy as np

# for x in chain(i1, i2, ...) loops over i1, then i2, then ...
# for x in repeat(value, n) gives x = value, n times.
from itertools import chain, repeat

from collections import Counter

from Point import Point
from Tensor import TensorLike, Tensor
from MachineLearning import MachineLearning
from Optimizer import Optimizer


class NeuralNetwork:
    """
    General NN structure which allows greater flexibility
    (and neuroevolution?).
    """

    nodes: Sequence[NeuralNode]
    """The NeuralNode's stored in the neural network."""

    input_nodes: Sequence[NeuralNode]
    """Nodes which attain values through input."""

    hidden_nodes: Sequence[NeuralNode]
    """Nodes which are not interacted with externally."""

    output_nodes: Sequence[NeuralNode]
    """Nodes used to get outputs."""

    def __init__(
            self,
            node_connections: Dict[int, List[int]],
            activation: Dict[int, Callable[[float], float]] = {},
            optimizer: Optimizer = Optimizer(),
    ):
        """Convert input integers to actual nodes, with activation functions appropriately."""

        # convert dict to list
        if isinstance(node_connections, dict):
            node_connections = [node_connections[key] for key, _ in enumerate(node_connections)]

        # create nodes
        self.nodes = [NeuralNode() for _ in node_connections]
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []

        # count how many outputs each node has
        output_counter = Counter(chain.from_iterable(node_connections))
        output_counter = [output_counter[i] for i, _ in enumerate(self.nodes)]

        # count amount of weights and biases
        weights = sum(len(connections) for connections in node_connections)
        biases = sum(1 for connections in node_connections if len(connections) > 0)

        # add weights and biases to ml object
        self.ml = MachineLearning(Point(Tensor([
            Tensor.random((weights+biases,)),
            Tensor.zeros((weights+biases,)),
        ])), optimizer)
        items = self.ml.items

        # cursor for tracking what to give out to neural nodes
        ml_cursor = 0

        # assign input weights and biases, and initialize output weights
        for outputs, node, connections in zip(output_counter, self.nodes, node_connections):

            # initialize output weights
            node.next_weights = []

            # assign input neurons
            node.previous_outputs = [self.nodes[index] for index in connections]

            # assign input weights
            node.previous_weights = Point(items[:, ml_cursor:ml_cursor+len(connections)])
            ml_cursor += len(connections)

            # assign output neurons and weights
            for previous_neuron, previous_weight in zip(node.previous_outputs, node.previous_weights):
                previous_neuron.next_inputs.append(node)
                previous_neuron.next_weights.append(previous_weight)

            if len(connections) > 0:

                # assign biases
                node.bias = items[:, ml_cursor]

                ml_cursor += 1

        for node in self.nodes:

            # cast output weights to tensor
            node.next_weights = Point(Tensor(node.next_weights))

            # collect input/output values from connected nodes
            node.previous_outputs = Point(Tensor(
                zip(*[neuron.neuron_output for neuron in node.previous_outputs])))
            node.next_inputs = Point(Tensor(
                zip(*[neuron.neuron_input for neuron in node.next_inputs])))

            # assign to {input/hidden/output} nodes
            if len(node.next_inputs) == len(node.previous_outputs) == 0:
                raise ValueError("Unconnected node with no input or output connections")
            elif len(node.previous_outputs) == 0:
                self.input_nodes.append(node)
            elif len(node.next_inputs) == 0:
                self.output_nodes.append(node)
            else:
                self.hidden_nodes.append(node)

        # Assign activation functions
        for key, func in activation.items():
            self.nodes[key].activation = func

        if len(self.input_nodes) == 0:
            raise ValueError("No input nodes")

        if len(self.output_nodes) == 0:
            raise ValueError("No output nodes")


    def __call__(
            self,
            input_values: TensorLike,
            pad: float = 0,
    ) -> List[float]:
        """
        Fill in node.value's with the input_values,
        fill in remaining nodes with the pad value,
        perform feed-forward propagation,
        and return the node.value's from the output_nodes.
        """

        # flatten the input
        input_values = self.numpy_flatten(input_values)

        if len(input_values) > len(self.input_nodes):
            raise ValueError(
                f"Too many inputs, expected at most {len(self.input_nodes)} nodes")

        # fill in the input, followed by padded values
        input_nodes = iter(self.input_nodes)
        for node, value in zip(input_nodes, input_values):
            node.neuron_output.values = value
        for node in input_nodes:
            node.neuron_output.values = pad

        # perform feed-forward propagation over the hidden then output nodes
        for node in self.hidden_nodes:
            node.feed_forward()
        for node in self.output_nodes:
            node.feed_forward()

        return [float(node.neuron_output.values) for node in self.output_nodes]


    def train(
            self,
            input_data: Sequence[TensorLike],
            *output_data: Tuple[Sequence[TensorLike], ...],
            pad: float = 0,
            epochs: int = 1000,
            batch_size: int = 10,
    ):
        """
        Train is used to take a data set and train it based on
        an input and output.
        """

        # invalid input: model.train([X, Y, Z, ...])
        if output_data == () and len(input_data[0]) != 2:
            raise ValueError("Expected model.train([X, Y]) or model.train(X, Y). "
                             "Got model.train([X, Y, ...]) instead.")

        # invalid input: model.train(X, Y, Z, ...)
        elif len(output_data) > 1:
            raise ValueError("Expected model.train([X, Y]) or model.train(X, Y). "
                             "Got model.train(X, Y, ...) instead.")

        # modify model.train([X, Y]) to model.train(X, Y)
        elif output_data == ():
            input_data, output_data = input_data

        if len(input_data) != len(output_data):
            raise ValueError("Not the same amount of inputs and outputs given")

        # flatten and pad inputs
        for i, x in enumerate(input_data):
            x = self.numpy_flatten(x)

            # check amount of inputs
            if len(x) > len(self.input_nodes):
                raise ValueError(
                    f"Too many inputs, expected at most {len(self.input_nodes)} nodes")
            elif len(x) < len(self.input_nodes):
                x = np.array(list(x) + [pad] * (len(self.input_nodes) - len(x)))

            input_data[i] = x

        # flatten and pad outputs
        for i, y in enumerate(output_data):
            y = self.numpy_flatten(y)

            # check amount of outputs
            if len(y) > len(self.output_nodes):
                raise ValueError(
                    f"Too many outputs, expected at most {len(self.output_nodes)} nodes")
            elif len(y) < len(self.output_nodes):
                y = np.array(list(y) + [pad] * (len(self.output_nodes) - len(y)))

            output_data[i] = y

        # store as numpy arrays
        input_data = np.array(input_data)
        output_data = np.array(output_data)
        indexes = np.array(range(len(input_data)))

        self.ml.iteration = 0
        batch_counter = 0
        while self.ml.iteration < epochs:

            # shuffle and loop through the data
            np.random.shuffle(indexes)
            for i in indexes:
                x = input_data[i]
                y = output_data[i]
                batch_counter = (batch_counter + 1) % batch_size

                # insert x into the model
                self(x, pad=pad)

                # adjust output derivatives from expected values
                for node, expected_value in zip(self.output_nodes, y):
                    node.neuron_output.derivatives += node.neuron_output.values - expected_value

                # end of current batch
                if batch_counter == 0:

                    self.back_propagate()
                    self.ml.optimize()

                    # stop after epoch many optimizations
                    if self.ml.iteration >= epochs:
                        break

                    # reset loss function
                    for node in self.output_nodes:
                        node.neuron_output.derivatives = 0


    def back_propagate(self):
        """Back-propagate derivatives, looping in reverse of forward propagation."""
        for node in reversed(self.output_nodes):
            node.back_propagate()
        for node in reversed(self.hidden_nodes):
            node.back_propagate()


    @staticmethod
    def numpy_flatten(x):
        """Flatten x as a numpy array."""

        # cast to numpy if needed
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # flatten if needed
        if len(x.shape) != 1:
            x = x.flatten()

        return x


class NeuralNode:
    """
    General Neural Node structure which allows greater flexibility
    (and neuroevolution?).
    """

    #=======================#
    # Basic node attributes #
    #=======================#

    neuron_output: Point
    """
    Scalar representing the value in the node
    after the activation function is applied.
    
    neuron_output = activation(neuron_input)
    """

    neuron_input: Point
    """
    Scalar representing the value in the node
    before the activation function is applied.
    
    neuron_input = np.dot(weights, input_neurons.neuron_output) + bias
    """

    bias: Point
    """bias = Tensor([value, derivative]) holds the value of the bias and its derivative."""

    activation: Callable[[float], float]
    """
    Scalar function representing the activation function.
    Defaulted to the identity function.
    """

    previous_outputs: Point
    """Vector of previous neuron outputs."""

    previous_weights: Point
    """Vector of weights for previous neurons."""

    next_inputs: Point
    """Vector of derivatives from the next neurons."""

    next_weights: Point
    """Vector of weights for next neurons."""


    def __init__(
        self,
        activation: Callable[[float], float] = lambda x: x,  # no change
    ):
        """
        Initialize with float (random),
        activation function (no change by default),
        and connected nodes (no connections),
        and random weights.
        """

        # node attributes
        self.neuron_output = Point(Tensor.zeros((2,)))
        self.neuron_input = Point(Tensor.zeros((2,)))
        self.activation = activation
        self.bias = None
        self.previous_outputs = []
        self.previous_weights = []
        self.next_inputs = []
        self.next_weights = []


    #=========#
    # Methods #
    #=========#


    def feed_forward(self):
        """Uses input nodes to compute self.value."""
        self.neuron_input.values = np.dot(self.previous_weights.values, self.previous_outputs.values) + self.bias.values
        self.neuron_output.values = self.activation(self.neuron_input.values)


    def back_propagate(self):
        """Updates derivatives of all parameters."""

        # d(loss) / d(output)
        if len(self.output_neurons.derivatives) > 0:
            self.neuron_output.derivatives = np.dot(self.next_weights.values, self.next_inputs.derivatives)

        # d(activation) / d(input)
        try:
            derivative = self.activation.derivative(self.neuron_input.values)
        except AttributeError:
            x = self.neuron_input.values
            dx = 1e-8 * (1 + abs(x))
            derivative = (self.activation(x+dx) - self.activation(x-dx)) / (2*dx)

        # d(loss) / d(input) = d(loss) / d(bias)
        self.neuron_input.derivatives = self.bias.derivatives = derivative * self.neuron_output.derivatives

        # d(loss) / d(weight)
        self.previous_weights.derivatives = self.bias.derivatives * self.previous_outputs.values
