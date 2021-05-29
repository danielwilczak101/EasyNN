from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from EasyNN import handler
from EasyNN.Model import Model
from EasyNN.Cost import Cost, CrossEntropy, LogLikelihood


@handler
class ReLU(Model):

    def ff(self: ReLU, values: np.ndarray) -> np.ndarray:
        """
        Relu function which maps each value to the greater number
        between 0 and the value in question.
        
        Example
        -------
        >>> relu(np.array([0, -2, 4]))
        array([0, 0, 4])
        """
        return np.maximum(values, 0)

    def ff_parallel(self: ReLU, values: np.ndarray) -> np.ndarray:
        """
        Relu function which maps each value to the greater number
        between 0 and the value in question.
        
        Example
        -------
        >>> relu(np.array([0, -2, 4]))
        array([0, 0, 4])
        """
        return np.maximum(values, 0)
      	
	def bp(self: ReLU, derivatives: np.ndarray) -> np.ndarray:
        """
        """
        


@handler
class LeakyReLU(Model):

    def ff(self: LeakyReLU, values: np.ndarray) -> np.ndarray:
        """
        Implements feedforward propagation using the formula:
            leaky_relu(x) = np.maximum(x, 0.01*x)

        Example
        -------
        >>> leaky_relu(np.array([-1, 2, 3, -4]))
        array([-0.01,  2.  ,  3.  , -0.04])
        """
        return np.maximum(values, 0.01*values)

    def bp(self: LeakyReLU, derivatives: np.ndarray) -> np.ndarray:
        return ...


@handler
@dataclass
class Logistic(Model):
    cost: Cost = CrossEntropy

    def ff(self: Logistic, values: np.ndarray) -> np.ndarray:
        """
        Implements feedforward propagation using the formula:
            logistic(x) = 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-values))

    def ff_parallel(self: Logistic, values: np.ndarray) -> np.ndarray:
        """
        Implements feedforward propagation using the formula:
            logistic(x) = 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-values))

    def bp(self: Logistic, derivatives: np.ndarray) -> np.ndarray:
        """
        Implements back propagation using the formula:
            logistic.backpropagate(derivatives) = (1 - logistic(x)) * logistic(x)
        """
        return (1 - self.output.values) * self.output.values

    def bp_parallel(self: Logistic, derivatives: np.ndarray) -> np.ndarray:
        """
        Implements back propagation using the formula:
            logistic.backpropagate(derivatives) = (1 - logistic(x)) * logistic(x)
        """
        return (1 - self.output.values) * self.output.values
