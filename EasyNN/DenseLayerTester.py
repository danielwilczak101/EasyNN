
import DenseLayer

if __name__ == '__main__':
    inputLayer = DenseLayer.DenseLayer(10) #input values
    middleLayer = DenseLayer.DenseLayer(100, previousLayer=inputLayer) #middle value
    outputLayer = DenseLayer.DenseLayer(10, previousLayer=middleLayer) #output Layer
    inputLayer.output = [10, 5, 3, 2, 1, 0, 80, 60, 10, 20]
    middleLayer.forward()
    outputLayer.forward()
    print("middle: " + str(middleLayer.output))
    print("output:" + str(outputLayer.output))
