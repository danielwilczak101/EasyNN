import ml_optimizer_test as opt_tester
import os, sys, argparse
from multiprocessing import Pool
from typing import Optional, Tuple
from numpy import pi, exp, sin, cos, sqrt
#no bounds


class OptimizerTest:
    """Interface for optimizer test cases."""
    dimensions: int = 2
    bounds: Optional[Tuple[float, float]] = (-10, +10)
    initial_point: Optional[Tuple[float, ...]] = None
    iterations: int = 1000
    learning_rate: float = 3e-3
    plot_density: int = 25

    def __init__(self, name = ""):
        self.name = type(self).__name__ + str(name)

    @staticmethod
    def func(*args: Tuple[float, ...]):
        pass

    @staticmethod
    def derivative(*args: Tuple[float, ...]):
        pass


class Sphere(OptimizerTest):
    """Test case for the sphere function."""
    learning_rate = 0.1

    @staticmethod
    def func(x: float, y: float):
        return (x**2 + y**2) / 100

    @staticmethod
    def derivative(x: float, y : float):
        return 0.02*x, 0.02*y

    
class Rastrigin(OptimizerTest):
    """Test case for the Rastrigin function."""

    @staticmethod
    def func(x: float, y: float):
        f = lambda t: t**2 + 10 * (1 - cos(2*pi*t))
        return f(x) + f(y)

    @staticmethod
    def derivative(x: float, y : float):
        f = lambda t: 2*t + 20 * pi * sin(2*pi*t)
        return (f(x), f(y))


class Ackley(OptimizerTest):
    """Test case for the Ackley function."""

    @staticmethod
    def func(x: float, y: float):
        term1 = -20 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))
        term2 = -exp(0.5 * (cos(2*pi*x) + cos(2*pi*y)))
        return term1 + term2 + exp(1) + 20

    @staticmethod
    def derivative(x: float, y: float):
        sqrtsquares = sqrt(0.5 * (x**2 + y**2))
        term1 = 4 / sqrtsquares * exp(-0.2 * sqrtsquares)
        term2 = pi * exp(0.5 * (cos(2*pi*x) + cos(2*pi*y)))
        f = lambda t: t * term1 + sin(2*pi*t) * term2
        return (f(x), f(y))


def runTest(test_case: OptimizerTest, outputDir: str = 'output'):
    #grab, and create directory if it doesnt exist
    directory = os.path.isdir(outputDir)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    filePath = outputDir + "/" + test_case.name
    if test_case.dimensions == 2:
        opt_tester.plot_3D_test(
            file_name=filePath,
            func=test_case.func,
            derivative=test_case.derivative,
            bounds=test_case.bounds,
            initial_point=test_case.initial_point,
            iterations=test_case.iterations,
            learning_rate=test_case.learning_rate,
            plot_density=test_case.plot_density,
        )

    else:
        raise ValueError("Only test cases with 2 inputs are implemented.")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Runs tests on optimizers, if the process itself keeps failing, run with the -s or --single-thread option")
	parser.add_argument('-s', '--singleThread', help='Runs all tests in single threads, safer but slower', action='store_true')
	args = parser.parse_args()
	if args.singleThread:
		runTest(Sphere())
		runTest(Rastrigin())
		runTest(Ackley())
	else:
		with Pool(3) as p:
			p.map(runTest, [Sphere(), Rastrigin(), Ackley()])
