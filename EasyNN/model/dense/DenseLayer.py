from __future__ import annotations
import numpy as np
from EasyNN.ml_data_structure import Point

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
			self.weights = Point.Point(shape=(previousLayer.output.points.size, n_neurons))
			self.weights.randomize()
			self.biases = Point.Point(shape=(n_neurons,))
			self.biases.randomize()
			self.neuron_output = Point.Point(shape=(1, n_neurons))
		self.output = Point.Point(shape=(1, n_neurons))
        # There are two outputs because a layer has a activation function however for back propogation we need pre and
        # post activation values

	def forward(self):
		"""
		Forward pass of the neural Network, assumes previousLayer.output contains the correct values
		"""
		if self.previousLayer is not None:
			self.neuron_output.points = np.dot(self.previousLayer.output.points, self.weights.points) + self.biases.points
			self.output.points = self.activationFunction(self.neuron_output.points)
			print("OUTPUT: " + str(self.output))

	def backward(self):
        #Gradiant in is a 1d array that has the gradiant value for each neuro
		self.d_inputs = []
		if self.previousLayer is not None:
			d_activation = self.activationFunction.derivative(self.neuron_output.points)
			print("d_activation: " + str(d_activation))
			print("neuron_output: " + str(self.neuron_output))
			
			self.neuron_output.derivatives = self.output.derivatives * d_activation
			self.previousLayer.output.derivatives = np.dot(self.neuron_output.derivatives, self.weights.points.T)
			self.weights.derivatives = np.dot(self.previousLayer.output.points.T, self.neuron_output.derivatives)
        def getItems(self, items):
                values = self.weights.points.flatten()
                values = np.append(values, self.biases.points.flatten())
                
                derivatives = self.weights.derivatives.flatten()
                derivatives = np.append(values, self.biases.derivatives.flatten())
                return (values, derivatives)
                
        def setItems(self, items):
                pass
