from EasyNN.model.Model import Model
import numpy as np
from numpy.typing import ArrayLike
from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.optimizers.GradientDescent import GradientDescent
from EasyNN.Batch import Batch
from EasyNN.Batch import MiniBatch


class Weights(Model):
    """Generic model API."""
    parameters: np.ndarray


    def __init__(self, inputs: int, outputs: int) -> None:
        max_weight = np.sqrt(6 / (inputs + outputs))
        self.parameters = np.vstack([
            np.random.uniform(-max_weight, max_weight, inputs * outputs),
            np.zeros(inputs * outputs),
        ])
        self.shape = (outputs, inputs)


    def __call__(self, values: ArrayLike) -> np.ndarray:
        """
        Feedforward propagation of the values.

        Parameters
        ----------
        values : ArrayLike
            Takes an array-like input.

        Returns
        -------
        out : np.ndarray
            Weights.matrix @ values
        """
        self.inputs = np.array(values, copy=False)
        return self.matrix @ self.inputs


    def backpropagate(self, derivatives: ArrayLike) -> np.ndarray:
        """
        Back propagation of the derivatives.

        Parameters
        ----------
        derivatives : ArrayLike
            Takes an array-like input.

        Returns
        -------
        out : np.ndarray
            Weights.matrix.T @ derivatives
        """
        derivatives = np.array(derivatives, copy=False)

        if derivatives.ndim == 1:
            self.derivatives = np.outer(derivatives, self.inputs).flatten()

        else:
            self.derivatives = (derivatives.T @ self.inputs).flatten()
            self.derivatives /= len(derivatives)

        return self.matrix.T @ derivatives


    @property
    def matrix(self) -> np.ndarray:
        """Property for interpreting Weights.values as a matrix."""
        return self.values.reshape(self.shape)


    @matrix.setter
    def matrix(self, values: ArrayLike) -> None:
        """Replaces the values in the shape of the matrix."""
        self.matrix[...] = values
