from __future__ import annotations
import numpy as np


class DenseLayer:
	"""
	This is a dense layer, where all outputs of the previous layer are hooked up to all nodes of the current layer...
	This is a lot of linear algebra to get this working
	"""
	#if previousLayer is None, then this acts as an input layer
	def __init__(self, n_neurons: int, previousLayer: DenseLayer = None, activationFunction = None):
		"""
		Initialization of DenseLayer, if previousLayer is None, it assumes it is a input layer with no weights or biases
		"""
		self.previousLayer = previousLayer
		self.n_neurons = n_neurons
		if previousLayer is not None:
			self.activationFunction =  np.vectorize(activationFunction)
			self.activationFunction.derivative =  np.vectorize(activationFunction.derivative)
			self.weights = np.random.randn(previousLayer.output.size, n_neurons)
			self.biases = np.random.randn(n_neurons)
			self.neuron_output = np.zeros((1, n_neurons)) #This is pre activation
		self.output = np.zeros((1, n_neurons)) #this is post activation
        # There are two outputs because a layer has a activation function however for back propogation we need pre and
        # post activation values

	def forward(self):
		"""
		Forward pass of the neural Network, assumes previousLayer.output contains the correct values
		"""
		if self.previousLayer is not None:
			self.neuron_output = np.dot(self.previousLayer.output, self.weights) + self.biases
			self.output = self.activationFunction(self.neuron_output)
			print("OUTPUT: " + str(self.output))

	def backward(self, gradiantIn):
        #Gradiant in is a 1d array that has the gradiant value for each neuro
		self.d_inputs = []
		if self.previousLayer is not None:
			d_activation = self.activationFunction.derivative(self.neuron_output)
			gradiantCpy = gradiantIn.copy()
			print("d_activation: " + str(d_activation))
			print("neuron_output: " + str(self.neuron_output))
			gradiantCpy = np.multiply(gradiantCpy, d_activation)
			self.d_inputs = np.dot(gradiantCpy, self.weights.T)
			self.d_weights = np.dot(self.previousLayer.output.T, gradiantCpy)

