from EasyNN.model.Model import Model
import numpy as np
from numpy.typing import ArrayLike
from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.optimizers.GradientDescent import GradientDescent
from EasyNN.Batch import MiniBatch


class Weights(Model):
    """Generic model API."""
    parameters: np.ndarray
    

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
        return np.dot(values, self.values.T)


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
        self.derivatives = np.dot(derivatives, self.inputs.T).T #im too lazy to figure simplify
        return np.dot(derivatives, self.values)

    def train(
            self,
            inputs: ArrayLike,
            outputs: ArrayLike,
            epochs: int = 1000,
            optimizer: Optimizer = GradientDescent,
            batches: Batch = MiniBatch(8),
            loss = None,  # implement later
        ) -> None:
        """
        Trains the model using inputs and outputs.

        Parameters
        ----------
        inputs : ArrayLike
            Values plugged into the model.
        outputs : ArrayLike
            Expected outputs from the model.
        epochs : int = 10000
            Number of times the entire dataset is passed through.
        optimizer : Optimizer
            The optimizer used to update the parameters.
        batches : Batch = MiniBatch(8)
            How batches are extracted from the dataset.
        loss : TBA
            The cost function used for computing loss error and
            derivatives to be backpropagated from.

        Raises
        ------
        NotImplementedError : Needs to be overridden by subclass.
        """
        raise NotImplementedError #TODO

