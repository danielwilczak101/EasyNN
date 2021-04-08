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
    def __init__(self, num_inputs, num_neurons):
        self.parameters = np.random.random_sample((2, num_neurons, num_inputs))
        self.num_inputs = num_inputs
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
        self.inputs = values #need to save for back propogation
        if len(np.array(values).shape) == 2: #save number of batches for later
            self.numBatches = np.array(values).shape[0]
        else:
            self.numBatches = 1
            
        fwd = np.dot(values, np.array(self.values).T)
        if self.numBatches == 1:
            return fwd.flatten() #flatten again
        else:
            return fwd

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
        self.derivatives = np.dot(np.matrix(self.inputs).T, np.matrix(derivatives)).T #im too lazy to figure simplify
        back = np.dot(derivatives, np.array(self.values))

        if self.numBatches == 1:
            return back.flatten()
        else:
            return back

