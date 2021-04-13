import numpy as np
import math
from numpy.typing import ArrayLike
from typing import Union, Iterable, Tuple, List
from EasyNN.models.Model import Model
from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.optimizers.GradientDescent import GradientDescent
from EasyNN.Batch import MiniBatch
from EasyNN.loss.Loss import Loss
from EasyNN.loss.MeanSquareError import MeanSquareError

class RELUActivation(Model):
    """ The step activation function"""
    def activationFunction(input: float) -> float:
        return 1 / (1 + math.exp(-x))
                    
    def d_activationFunction(input: float) -> float:
        return activationFunction(input) * (1 - activationFunction(input))
                    
    def __init__(self):
        self.activation = np.vectorize(activationFunction)
        self.d_activation = np.vectorize(d_activationFunction)
    
    def __call__(self, values: ArrayLike) -> np.ndarray:
        self.input = values
        return self.activation(values)

    def backpropogate(self, derivatives: ArrayLike) -> np.ndarray:
        return self.d_activation(input) * derivatives
    
