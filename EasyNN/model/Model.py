import numpy as np
from numpy.typing import ArrayLike
from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.optimizers.GradientDescent import GradientDescent


class Model:
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

        Raises
        ------
        NotImplementedError : Needs to be overridden by subclass.
        """
        raise NotImplementedError


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

        Raises
        ------
        NotImplementedError : Needs to be overridden by subclass.
        """
        raise NotImplementedError


    def train(
            self,
            inputs: ArrayLike,
            outputs: ArrayLike,
            optimizer: Optimizer = GradientDescent,
            loss_function = None,  # implement later
        ) -> None:
        """
        Trains the model using inputs and outputs.

        Parameters
        ----------
        inputs : ArrayLike
            Values plugged into the model.
        outputs : ArrayLike
            Expected outputs from the model.
        optimizer : Optional[Optimizer]
            The optimizer used to update the parameters.
        loss_function : TBA
            The cost function used for computing loss values and error
            derivatives to be backpropagated from.

        Raises
        ------
        NotImplementedError : Needs to be overridden by subclass.
        """
        raise NotImplementedError
