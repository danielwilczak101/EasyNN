import sys
sys.path.append("../") 
from EasyNN.model.Weights import Weights as Weights

def runWeightTest() -> bool:
    """
    Runs a test on all of the models

    Returns
    -----
    bool
        A bool that states whether the test passed or failed
    """
    success = True
    
    weights = weights.Weights()
    weights.values = [[1, 2], [2, 3]]
    forwardPass = weights([4,5])
    d_inputs = weights.backpropogate([6,7])
    d_weights = weights.derivatives

    expected_forwardPass = [ 14,32 ]
    expected_d_weights = [[24, 30], [28, 35]]
    expected_d_inputs = [27, 40]
    
    if expected_forwardPass != forwardPass:
        success = False
        print("Forward Pass failed")
        print("Expected: "+  str(expected_forwardPass))
        print("Actual: " + str(forwardPass))
    
    if expected_d_weights != d_weights:
        success = False
        print("d_weights incorrect")
        print("Expected: "+  str(expected_d_weights))
        print("Actual: " + str(d_weights))
    
    if expected_d_inputs != d_inputs:
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
    return runWeightTest()
    
if __name__ == '__main__':
    runTest()
