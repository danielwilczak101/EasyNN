import numpy as np
import DenseLayer

if __name__ == '__main__':
    inputLayer = DenseLayer.DenseLayer(2) #input values
    middleLayer = DenseLayer.DenseLayer(3, previousLayer=inputLayer) #middle value
    outputLayer = DenseLayer.DenseLayer(2, previousLayer=middleLayer) #output Layer
    inputLayer.output = np.array([[10, 5]])
    print(str(inputLayer.output))
    middleLayer.forward()
    outputLayer.forward()
    outputLayer.backward(np.array([[1, 2]]))
    middleLayer.backward(outputLayer.d_inputs)
    inputLayer.backward(middleLayer.d_inputs)

