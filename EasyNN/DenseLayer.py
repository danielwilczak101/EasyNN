from __future__ import annotations
import numpy as np


class DenseLayer:
    """
    This is a dense layer, where all outputs of the previous layer are hooked up to all nodes of the current layer...
    This is a lot of linear algebra to get this working
    """
    #if previousLayer is None, then this acts as an input layer
    def __init__(self, n_neurons: int, previousLayer: DenseLayer = None):
        """
        Initialization of DenseLayer, if previousLayer is None, it assumes it is a input layer with no weights or biases
        """
        self.previousLayer = previousLayer
        if previousLayer is not None:
            self.weights = 0.01 * np.random.randn(previousLayer.output.size, n_neurons)
            self.biases = 0.01 * np.random.randn(n_neurons)
            self.neuron_output = np.zeros((1, n_neurons)) #This is pre activation
        self.output = np.zeros((1, n_neurons)) #this is post activation
        # There are two outputs because a layer has a activation function however for back propogation we need pre and
        # post activation values

    def forward(self):
        """
        Forward pass of the neural Network, assumes previousLayer.output contains the correct values
        """
        self.neuron_output = np.dot(self.previousLayer.output, self.weights) + self.biases
        print("neuron_output" + str(self.neuron_output))
        #TODO change this later, rn it only does RELU
        self.output = np.maximum(0, self.neuron_output)
        print("output" + str(self.output))

    def backward(self, gradiantIn):
        #Gradiant in is a 1d array that has the gradiant value for each neuron
        print("\n\nBack prop start")
        if self.previousLayer is not None:
            print("weights:" + str(self.weights))
            print("inputs:" + str(self.previousLayer.output))
            gradiantCpy = gradiantIn.copy()
            print("Gradiant copy: " + str(self.neuron_output < 0))
            gradiantCpy[self.neuron_output < 0] = 0
            print("gradiantCpy: " + str(gradiantCpy))
            self.d_inputs = np.dot(gradiantCpy, self.weights.T)
            print("d_inputs:" + str(self.d_inputs))
            self.d_weights = np.dot(self.previousLayer.output.T, gradiantCpy)
            print("d_weights:" + str(self.d_weights))

