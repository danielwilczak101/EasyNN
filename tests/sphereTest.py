import ml_optimizer_test as opt_tester
import os, sys
from typing import Optional, Tuple
#no bounds


class OptimizerTest:
    """Interface for optimizer test cases."""

    dimensions: int = 2
    bounds: Optional[Tuple[float, float]] = (-10, +10)
    initial_point: Optional[Tuple[float, ...]] = None
    iterations: int = 1000
    learning_rate: float = 0.1

    @staticmethod
    def func(*args: Tuple[float, ...]):
        pass

    @staticmethod
    def derivative(*args: Tuple[float, ...]):
        pass


class Sphere(OptimizerTest):
    """Test case for sphere function."""

    @staticmethod
    def func(x: float, y: float):
        return (x**2 + y**2) / 100

    @staticmethod
    def derivative(x: float, y : float):
        return 0.02*x, 0.02*y


def runTest(test_case: OptimizerTest, outputDir: str = 'output'):
    #grab, and create directory if it doesnt exist
    directory = os.path.isdir(outputDir)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    if test_case.dimensions == 2:
        opt_tester.plot_3D_test(
            file_name=test_case.__name__,
            func=test_case.func,
            derivative=test_case.derivative,
            bounds=test_case.bounds,
            initial_point=test_case.initial_point,
            iterations=test_case.iterations,
            learning_rate=test_case.learning_rate,
        )

    else:
        raise ValueError("Only test cases with 2 inputs are implemented.")

if __name__ == '__main__':
    runTest(Sphere)
