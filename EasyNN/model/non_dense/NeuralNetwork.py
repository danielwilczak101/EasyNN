"""
Generic Neural Network module. Use NeuralNetwork() to create a neural network.
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
        self.ml = MachineLearning(Tensor.random((2, weights+biases)), optimizer)
        items = self.ml.items

        # cursor for tracking what to give out to neural nodes
        ml_cursor = 0

        # assign input weights and biases, and initialize output weights
        for outputs, node, connections in zip(output_counter, self.nodes, node_connections):

            # initialize output weights
            node.next_weights = []

            # assign input neurons
            node.previous_neurons = [self.nodes[index] for index in connections]

            # assign input weights
            node.previous_weights = items[:, ml_cursor:ml_cursor+len(connections)]
            ml_cursor += len(connections)

            # assign output neurons and weights
            for previous_neuron, previous_weight in zip(node.previous_neurons, node.previous_weights):
                previous_neuron.next_neurons.append(node)
                previous_neuron.next_weights.append(previous_weight)

            if len(connections) > 0:

                # assign biases
                node.bias = items[:, ml_cursor]

                ml_cursor += 1

        for node in self.nodes:

            # cast output weights to tensor
            node.next_weights = Tensor(node.next_weights)

            # collect input/output values from connected nodes
            node.previous_neurons = Tensor(
                [neuron.neuron_output[0] for neuron in node.previous_neurons])
            node.next_neurons = Tensor(
                [neuron.bias[1] for neuron in node.next_neurons])

            # assign to {input/hidden/output} nodes
            if len(node.next_neurons) == len(node.previous_neurons) == 0:
                raise ValueError("Unconnected node with no input or output connections")
            elif len(node.previous_neurons) == 0:
                self.input_nodes.append(node)
            elif len(node.next_neurons) == 0:
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
    ) -> TensorLike:
        """
        Fill in node.value's with the input_values,
        fill in remaining nodes with the pad value,
        perform feed-forward propagation,
        and return the node.value's from the output_nodes.
        """

        # cast to numpy array
        if not isinstance(input_values, np.ndarray):
            input_values = np.array(input_values)

        # get the output shape and flatten the input
        input_values = input_values.flatten()

        if len(input_values) > len(self.input_nodes):
            raise ValueError(
                f"Too many inputs, expected at most {len(self.input_nodes)} nodes")

        # fill in the input, followed by padded values
        input_nodes = iter(self.input_nodes)
        for node, value in zip(input_nodes, input_values):
            node.neuron_output[0] = value
        for node in input_nodes:
            node.neuron_output[0] = pad

        # perform feed-forward propagation over the hidden then output nodes
        for node in self.hidden_nodes:
            node.feed_forward()
        for node in self.output_nodes:
            node.feed_forward()

        return [node.neuron_output[0] for node in self.output_nodes]


    def train(
            self,
            input_data: Sequence[TensorLike],
            *output_data: Tuple[Sequence[TensorLike], ...],
            epochs: int = 1000,
            batch_size: int = 10,
    ):
        """
        Train is used to take a data set and train it based on
        an input and output.
        """

        # merge input and output data
        if output_data != ():
            # flatten the data
            flatten = lambda tensor: np.reshape(tensor, (len(tensor),))
            input_data = map(flatten, input_data)
            output_data = map(flatten, output_data)

            # merge the data
            data = np.array(list(zip(input_data, output_data)))
        else:
            data = np.array(input_data)

        self.ml.iteration = 0
        batch_counter = 0
        while True:

            # shuffle and loop through the data
            np.random.shuffle(data)
            for x, y in data:
                batch_counter = batch_counter % batch_size + 1

                # insert x into the model
                self(x)

                # adjust output derivatives from expected values
                for node, expected_value in zip(self.output_nodes, y):
                    delta_output = node.neuron_output[0] - expected_value - node.neuron_output[1]
                    node.neuron_output[1] += delta_output / batch_counter

                if batch_counter == batch_size:

                    # back-propagate the derivative
                    for node in self.output_nodes:
                        node.back_propagate()
                    for node in self.hidden_nodes:
                        node.back_propagate()

                    # optimize parameters using chosen machine learning algorithm
                    self.ml.optimize()

                    # stop after epoch many optimizations
                    if self.ml.iteration >= epoch:
                        break

            # if for loop doesn't break, go again
            else:
                continue

            # if for loop breaks, break the while loop
            break


class NeuralNode:
    """
    General Neural Node structure which allows greater flexibility
    (and neuroevolution?).
    """

    #=======================#
    # Basic node attributes #
    #=======================#

    neuron_output: Tensor
    """
    Scalar representing the value in the node
    after the activation function is applied.
    
    neuron_output = activation(neuron_input)
    """

    neuron_input: Tensor
    """
    Scalar representing the value in the node
    before the activation function is applied.
    
    neuron_input = np.dot(weights, input_neurons.neuron_output) + bias
    """

    bias: Tensor
    """bias = Tensor([value, derivative]) holds the value of the bias and its derivative."""

    activation: Callable[[float], float]
    """
    Scalar function representing the activation function.
    Defaulted to the identity function.
    """

    previous_neurons: Sequence[NeuralNode]
    """Vector of previous neurons."""

    previous_weights: Tensor
    """Vector of weights/derivatives for previous nodes."""

    next_neurons: Sequence[NeuralNode]
    """Vector of next neurons."""

    next_weights: Tensor
    """Vector of weights/derivatives for next nodes."""


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
        self.neuron_output = Tensor.zeros((2,))
        self.neuron_input = Tensor.zeros((2,))
        self.activation = activation
        self.bias = None
        self.previous_neurons = []
        self.previous_weights = []
        self.next_neurons = []
        self.next_weights = []


    #=========#
    # Methods #
    #=========#


    def feed_forward(self):
        """Uses input nodes to compute self.value."""
        self.neuron_input[0] = np.dot(self.previous_weights[0], self.previous_neurons) + self.bias[0]
        self.neuron_output[0] = self.activation(self.neuron_input[0])


    def back_propagate(self):
        """Updates derivatives of all parameters."""

        # d(loss) / d(output)
        if len(self.output_neurons) > 0:
            self.neuron_output[1] = np.dot(self.next_weights, self.next_neurons)

        # d(activation) / d(input)
        try:
            derivative = self.activation.derivative(self.neuron_input[0])
        except AttributeError:
            x = self.neuron_input[0]
            dx = 1e-8 * (1 + abs(x))
            derivative = (self.activation(x+dx) - self.activation(x-dx)) / (2*dx)

        # d(loss) / d(input) = d(loss) / d(bias)
        self.neuron_input[1] = self.bias[1] = derivative * self.neuron_output[1]

        # d(loss) / d(weight)
        self.previous_weights[1] = self.bias[1] * self.previous_neurons
