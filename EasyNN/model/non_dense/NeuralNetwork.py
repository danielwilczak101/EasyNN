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

from Point import PointList as Point
from Point import TensorType


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
            nodes: Dict[int, List[int]],
            activation: Dict[int, Callable[[float], float]] = {},
    ):
        """
        Convert input integers to actual nodes, with activation functions
        appropriately.
        """

        # Convert dict to list
        if isinstance(nodes, dict):
            nodes = [nodes[key] for key, _ in enumerate(nodes)]

        # Create nodes
        self.nodes = [NeuralNode() for _ in nodes]

        # Create connections between nodes and assign weights
        for node, node_indexes in zip(self.nodes, nodes):
            node.input_nodes = [self.nodes[i] for i in node_indexes]
            node.input_weights = [Point() for _ in node_indexes]
            for i, weight in zip(node_indexes, node.input_weights):
                self.nodes[i].output_nodes.append(node)
                self.nodes[i].output_weights.append(weight)

        # Assign activation functions
        for key, func in activation.items():
            self.nodes[key].activation = func

        # Collect input nodes i.e. connecting to nothing
        self.input_nodes = []
        for node in self.nodes:
            if len(node.input_nodes) == 0:
                self.input_nodes.append(node)
            else:
                break

        # if no break
        else:
            raise ValueError("All nodes are input nodes")

        # Collect indexes of all nodes that are not outputs
        nodes_connected_from = {
            index
            for node_indexes in nodes
            for index in node_indexes
        }

        # Count valid output nodes
        amount_of_outputs = 0
        for index in range(len(nodes)-1, -1, -1):
            if index in nodes_connected_from:
                break
            else:
                amount_of_outputs += 1

        # Raise an error if there's no output
        if amount_of_outputs == 0:
            raise ValueError("No output nodes")

        # Collect output nodes
        self.output_nodes = self.nodes[-amount_of_outputs:]

        # Collect hidden nodes
        self.hidden_nodes = self.nodes[len(self.input_nodes):(len(self.nodes)-len(self.output_nodes))]

        # Randomize weights
        for node in self.output_nodes + self.hidden_nodes:
            for weight in node.input_weights:
                weight.randomize()


    def __call__(
            self,
            input_values: TensorType,
            output_shape: Union[Sequence[int], None] = None,
            pad: float = 0,
    ) -> TensorType:
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
        output_shape = output_shape or input_values.shape
        input_values = input_values.flatten()

        if len(input_values) > len(self.input_nodes):
            raise ValueError(
                f"Too many inputs, expected at most {len(self.input_nodes)} nodes")

        # fill in the inputs followed by repeated pads
        input_values = chain(
            input_values,
            repeat(pad, len(self.input_nodes) - len(input_values))
        )

        for node, value in zip(self.input_nodes, input_values):
            node.value.points = value

        # perform feed-forward propagation over the hidden then output nodes
        for node in self.hidden_nodes + self.output_nodes:
            node.feed_forward()

        # return the output in the original shape
        return np.reshape(
            [node.value.points for node in self.output_nodes],
            output_shape,
        )


    def train(
            self,
            input_data: Sequence[TensorType],
            output_data: Sequence[TensorType],
            epochs: int = 1000,
    ):
        """
        Train is used to take a data set and train it based on
        an input and output.
        """

        # flatten the data
        flatten = lambda tensor: np.reshape(tensor, (len(tensor),))
        input_data = map(flatten, input_data)
        output_data = map(flatten, output_data)

        # merge the data
        data = np.array(list(zip(input_data, output_data)))

        for i in range(epochs):

            # shuffle and loop through the data
            np.random.shuffle(data)
            for x, y in data:

                # insert x into the model
                self(x)

                # adjust output derivatives from expected values
                for node, expected_value in zip(self.output_nodes, y):
                    node.value.derivatives = 2*(node.value.points - expected_value)

                # back-propagate the derivative
                for node in reversed(self.nodes):
                    node.back_propagate()

                # update parameters based on derivatives
                for node in self.nodes:
                    node.optimize()


class NeuralNode:
    """
    General Neural Node structure which allows greater flexibility
    (and neuroevolution?).
    """

    #=======================#
    # Basic node attributes #
    #=======================#

    value: Point
    """
    Scalar representing the value in the node
    after the activation function is applied.
    
    value = activation(prevalue)
    """

    prevalue: Point
    """
    Scalar representing the value in the node
    before the activation function is applied.
    
    value = np.dot(weights, previous_nodes.value) + bias
    """

    bias: Point
    """Scalar representing the bias of the node."""

    activation: Callable[[float], float]
    """
    Scalar function representing the activation function.
    Defaulted to the identity function.
    """

    input_nodes: Sequence[NeuralNode]
    """Vector of input nodes."""

    input_weights: Sequence[Point]
    """Vector of weights for input nodes."""

    output_nodes: Sequence[NeuralNode]
    """Vector of output nodes."""

    output_weights: Sequence[Point]
    """Vector of weights for output nodes."""


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
        self.value = Point()
        self.prevalue = Point()
        self.bias = Point()
        self.activation = activation
        self.input_nodes = []
        self.input_weights = []
        self.output_nodes = []
        self.output_weights = []


    #=========#
    # Methods #
    #=========#


    def feed_forward(self):
        """Uses input nodes to compute self.value."""

        self.prevalue.points = sum(
            weight.points * node.value.points
            for weight, node
            in zip(self.input_weights, self.input_nodes)
        )

        self.value.points = self.activation(self.prevalue.points)


    def back_propagate(self):
        """Updates derivatives of all parameters."""

        # d(loss) / d(value)
        if len(self.output_nodes) > 0:
            self.value.derivatives = sum(
                weight.points * node.value.derivatives
                for weight, node
                in zip(self.output_weights, self.output_nodes)
            )

        # d(activation) / d(prevalue)
        try:
            derivative = self.activation.derivative(self.prevalue.points)
        except AttributeError:
            x = self.prevalue.points
            dx = 1e-8 * (1 + abs(x))
            derivative = (self.activation(x+dx) - self.activation(x-dx)) / (2*dx)

        # d(loss) / d(prevalue) = d(loss) / d(bias)
        self.prevalue.derivatives = derivative * self.value.derivatives
        self.bias.derivatives = derivative * self.value.derivatives

        # d(loss) / d(weight)
        for weight, node in zip(self.input_weights, self.input_nodes):
            weight.points = self.prevalue.derivatives * node.value.points


    def optimize(self):
        """Updates points based on their derivatives."""

        self.bias.optimize()

        for weight in self.input_weights:
            weight.optimize()
