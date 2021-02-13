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
            node.connected_nodes = [self.nodes[i] for i in node_indexes]
            node.weights = Point((len(node_indexes),))

        # Assign activation functions
        for key, func in activation.items():
            self.nodes[key].activation = func

        # Collect input nodes i.e. connecting to nothing
        self.input_nodes = []
        for node in self.nodes:
            if len(node.connected_nodes) == 0:
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

        # Cast to numpy array for reshaping
        if not isinstance(input_values, np.ndarray):
            input_values = np.array(input_values)

        # Get the output shape and flatten the input
        output_shape = output_shape or input_values.shape
        input_values = input_values.flatten()

        if len(input_values) > len(self.input_nodes):
            raise ValueError("Too many inputs, expected at most {len(self.input_nodes)} nodes")

        # Fill in the inputs followed by repeated pads
        input_values = chain(input_values, repeat(pad, len(self.input_nodes) - len(input_values)))

        for node, value in zip(self.input_nodes, input_values):
            node.value.points = [value]

        # Perform feed-forward propagation
        for node in chain(self.hidden_nodes, self.output_nodes):
            node.feed_forward()

        # Return the output in the original shape
        return np.array([
            node.value.points
            for node in self.output_nodes
        ]).reshape(output_shape)


    def train(
            self,
            input_data: TensorType,
            output_data: TensorType,
    ):
        """
        Train is used to take a data set and train it based on
        an input and output.
        """

        pass


class NeuralNode:
    """
    General Neural Node structure which allows greater flexibility
    (and neuroevolution?).
    """

    #=======================#
    # Basic node attributes #
    #=======================#

    value: Point
    """Scalar representing the value in the node."""

    bias: Point
    """Scalar representing the bias of the node."""

    activation: Callable[[float], float]
    """
    Scalar function representing the activation function.
    Defaulted to the identity function.
    """

    connected_nodes: Sequence[NeuralNode]
    """Vector of all connected nodes."""

    weights: Point
    """Vector of weights for connected nodes."""


    def __init__(
        self,
        activation: Callable[[float], float] = lambda x: x,  # no change
        connected_nodes: Sequence[NeuralNode] = tuple(),
    ):
        """
        Initialize with float (random by default),
        activation function (no change by default),
        and connected nodes (no connections by default)
        and random weights.
        Initialize delta's to 0.
        """

        # node attributes
        self.value = Point((1,))
        self.bias = Point((1,))
        self.activation = activation
        self.connected_nodes = connected_nodes
        self.weights = Point((len(connected_nodes),))


    #=========#
    # Methods #
    #=========#


    def feed_forward(self):
        """Uses connected nodes to compute self.value."""

        self.value.points = [self.activation(
            np.dot(
                self.weights.points,
                [node.value.points for node in self.connected_nodes]
            )
        )]


    def back_propagate(self):
        """
        Modify change attributes to more closely match value_change.
        Modify nodes.value_change to more closely match value_change.
        Modify respective attributes based on their attr_change.
        """
        pass
