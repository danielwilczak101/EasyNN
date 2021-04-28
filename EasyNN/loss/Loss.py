import numpy as np


class Loss:
    """Base loss class API."""

    def cost(
            self,
            inputs: np.ndarray,
            expectations: np.ndarray,
            predictions: np.ndarray
        ) -> float:
        """
        Computes the cost based on the inputs, expected results, and
        predicted results.

        Parameters
        ----------
        inputs : np.ndarray
            The inputs passed into the model being measured.
        expectations : np.ndarray
            The expected outputs of the model from the inputs.
        predictions : np.ndarray
            The predicted outputs from the model from the inputs.

        Returns
        -------
        float : The cost/error of the model.

        Raises
        ------
        NotImplementedError : Needs to be overriden by subclass.
        """
        raise NotImplementedError

    def gradient(
            self,
            inputs: np.ndarray,
            expectations: np.ndarray,
            predictions: np.ndarray
        ) -> np.ndarray:
        """
        Computes the gradient of the cost function.

        Parameters
        ----------
        inputs : np.ndarray
            The inputs passed into the model being measured.
        expectations : np.ndarray
            The expected outputs of the model from the inputs.
        predictions : np.ndarray
            The predicted outputs from the model from the inputs.

        Returns
        -------
        np.ndarray : The gradient of the cost function.

        Raises
        ------
        NotImplementedError : Needs to be overriden by subclass.

        Notes
        -----
        Will be replaced by numerical estimate of the gradient in the future.
        """
        raise NotImplementedError
