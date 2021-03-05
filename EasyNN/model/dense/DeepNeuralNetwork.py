import numpy as np
#import nnfs
#from nnfs.datasets import spiral_data
import EasyNN.model.dense.DenseLayer as DenseLayer
import EastNN.ml_data_structure.MachineLearning as ML
import EasyNN.ml_data_structure.Point as Point
class DeepNeuralNetwork:
	"""Class that is used to control the Deep Neural Network."""
        
	def __init__(self, numInputs, layerSpecification):
		"""
		layerSpecification: specifies each layer: [(numNeurons, activationFunction), ...]
		last layer specification is the output layer
		"""
		self.layers = []
		firstLayer = DenseLayer.DenseLayer(numInputs)
		previousLayer=firstLayer
                ml = ML.MachineLearning
		for numNeurons, activationFunction in layerSpecification:
			currentLayer = DenseLayer.DenseLayer(numNeurons, previousLayer=previousLayer, activationFunction=activationFunction)
			self.layers.append(currentLayer)
			previousLayer=currentLayer

                        
	def forward(self, inputValues):
		"""
		should be a list of inputs each of which is the size of layers[0].size		
		"""
		self.layers[0].output.points = inputValues #ya.... not the most intuitive statement....
		for layer in self.layers:
			layer.forward()
                        
		return self.layers[-1].output

	def backward(self, correctValues):
		"""
    	        should be a list of inputs each of which is the size of layers[0].size
		"""
		correctValues = np.array(correctValues)
		if len(correctValues.shape) == 1: #Change to 1 hot vectors if necessary
			correctValues = np.eye(self.layers[-1].n_neurons)[correctValues]
		        
		print(str(correctValues))
		#backPropogateValues = correctValues
		self.layers[-1].output.derivatives = correctValues
		for layer in reversed(self.layers):
			print(" LAYER started")
			layer.backward()

