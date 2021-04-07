import numpy as np
from EasyNN.loss.Loss import Loss


class MeanSquareError(Loss):
    """Most commonly used loss, the mean square error, uses the L2 norm squared."""

    def cost(
            self,
            inputs: np.ndarray,
            expectations: np.ndarray,
            predictions: np.ndarray
        ) -> float:
        """
        Computes the cost from the formula
        >>> norm(expect - predict) ** 2 / expect.size

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
        """
        return np.linalg.norm(expectations - predictions) ** 2 / expectations.size

    def gradient(
            self,
            inputs: np.ndarray,
            expectations: np.ndarray,
            predictions: np.ndarray
        ) -> np.ndarray:
        """
        Computes the gradient from the formula
        >>> 2 * (expect - predict) / expect.size

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
        """
        return (2 / expectations.size) * (expectations - predictions)
