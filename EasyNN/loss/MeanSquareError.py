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
            norm(expect - predict) ** 2 / (2 * batch_size)

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
        out : float
            The cost/error of the model.
        """
        return np.linalg.norm(expectations - predictions) ** 2 / (2 if expectations.ndim==1 else 2*len(expectations))

    def gradient(
            self,
            inputs: np.ndarray,
            expectations: np.ndarray,
            predictions: np.ndarray
        ) -> np.ndarray:
        """
        Computes the gradient from the formula
        >>> (expect - predict) / batch_size

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
        out : np.ndarray
            The gradient of the cost function.
        """
        if expectations.ndim == 1:
            return expectations - predictions
        else:
            return (expectations - predictions) / expectations.shape[0]
