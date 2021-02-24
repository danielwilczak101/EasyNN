import numpy as np
import DenseLayer
import DeepNeuralNetwork

def relu(x):
	if x > 0:
		return x
	else:
		return 0
		
def d_relu(x):
	if x > 0:
		return 1
	else:
		return 0
relu.derivative = d_relu

if __name__ == '__main__':
	layerSpecification = [(3, relu), (2, relu)]
	NN = DeepNeuralNetwork.DeepNeuralNetwork(2, layerSpecification)
	
	print(str(NN.forward([[10, 5]])))
    
	NN.backward([1])
	print(str(NN.layers[1].d_inputs))

