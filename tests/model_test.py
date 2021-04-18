import sys
sys.path.append("../") 
from EasyNN.model import Weights
import numpy as np

def runWeightTest() -> bool:
    """
    Runs a test on all of the models

    Returns
    -----
    bool
        A bool that states whether the test passed or failed
    """
    success = True
    
    weights = Weights.Weights(2, 2)
    weights.matrix = [[1.0, 2.0], [3.0, 4.0]]
    forwardPass = weights([4.0,5.0])
    d_inputs = weights.backpropagate([6.0,7.0])
    d_weights = weights.derivatives.reshape(weights.shape)

    expected_forwardPass = [ 14.0,32.0 ]
    expected_d_weights = [[24.0, 30.0], [28.0, 35.0]]
    expected_d_inputs = [27.0, 40.0]
    
    if not np.array_equal(expected_forwardPass, forwardPass):
        success = False
        print("Forward Pass failed")
        print("Expected: "+  str(expected_forwardPass))
        print("Actual: " + str(forwardPass))
    
    if not np.array_equal(expected_d_weights, d_weights):
        success = False
        print("d_weights incorrect")
        print("Expected: "+  str(expected_d_weights))
        print("Actual: " + str(d_weights))
    
    if not np.array_equal(expected_d_inputs, d_inputs):
        success = False
        print("d_inputs incorrect")
        print("Expected: "+  str(expected_d_inputs))
        print("Actual: " + str(d_inputs))
    return success

def runTest() -> bool:
    """
    Runs a test on all of the models

    Returns
    -----
    bool
        A bool that states whether the test passed or failed
    """
    testPass = runWeightTest()
    if not testPass:
        print("TESTS FAILED")
    
if __name__ == '__main__':
    runTest()
