import ml_optimizer_test as opt_tester
import os, sys
#no bounds

def sphereFunction(x : float, y : float):
    return (x ** 2) + (y ** 2)

def sphere_derivative(x: float, y : float):
    return ( (2 * x), (2 * y))

def runTest(outputDir:str='output'):
    #grab, and create directory if it doesnt exist
    directory=os.path.isdir(outputDir)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    outputFile = outputDir + "/sphere"
    
    
    opt_tester.plot_3D_test(file_name=outputFile,
                 func=sphereFunction,
                 derivative=sphere_derivative)
    
                 
if __name__ == '__main__':
    runTest()
