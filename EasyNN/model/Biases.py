from EasyNN.model.Model import Model
import numpy as np
from numpy.typing import ArrayLike
from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.optimizers.GradientDescent import GradientDescent
from EasyNN.Batch import Batch
from EasyNN.Batch import MiniBatch


class Biases(Model):
    """Generic model API."""
    parameters: np.ndarray
    def __init__(self, num_neurons):
        self.parameters = np.random.random_sample((2, num_neurons))
        self.num_neurons = num_neurons

    def __call__(self, values: ArrayLike) -> np.ndarray:
        """
        Feedforward propagation of the values.

        Parameters
        ----------
        values : ArrayLike
            Takes an array-like input.

        Returns
        -------
        np.ndarray
            A numpy array representing the output, representing the result
            of the model on the given input.
        """
        return self.values + values

    def backpropagate(self, derivatives: ArrayLike) -> np.ndarray:
        """
        Back propagation of the derivatives.

        Parameters
        ----------
        derivatives : ArrayLike
            Takes an array-like input.

        Returns
        -------
        np.ndarray
            A numpy array representing the output derivative, representing
            how the previous input should be changed to get the desired
            change in output when using the model.
        """
        self.derivatives = derivatives
        return derivatives
