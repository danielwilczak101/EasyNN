"""
TODO: Not complete.
"""
from __future__ import annotations
import numpy as np
from EasyNN.loss.abc import Loss
from EasyNN.typing import Array1D
import EasyNN.model.abc


class NegativeLikelihood(Loss):
    """
    Computes the loss as the log-likelihood of the desired result.

    Similar to the LogLikelihood, except doesn't take the log(). Leaves
    the log() step to the LogSoftMax, instead of using the SoftMax.

    Note: This should be used with the LogSoftMax activation as the last layer.
    """

    def __call__(self: NegativeLikelihood, x: Array1D, y: Array1D) -> float:
        # Check the inputs.
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if y.ndim > 2:
            raise IndexError(
                "y must be either 0-D (the index of a single sample),\n"
                "    or 1-D (multiple indexes or an array of probabilities),\n"
                "    or 2-D (multiple arrays of probabilities),\n"
                "    but not {y.ndim}-D"
            )
        # Extract the predicted probabilities.
        y_pred = self.model(x)
        # If y is an int, it represents the index of the probability being checked.
        if y.ndim == 0 and y_pred.ndim == 1:
            likelihoods = y_pred[int(y)]
        # If y is an array of indexes, check each index.
        elif y.ndim == 1 and y_pred.ndim == 2:
            likelihoods = y_pred[np.arange(len(y)), y.astype(int)]
        # If y is an array of probabilities, check the largest probability.
        elif y.ndim == 1 and y_pred.ndim == 1:
            likelihoods = y_pred[np.argmax(y.astype(int))]
        # If y is multiple arrays of probabilities, check the largest probability for each row.
        elif y.ndim == 2 and y_pred.ndim == 2:
            likelihoods = y_pred[np.arange(len(y)), np.argmax(y.astype(int), axis=1)]
        # Invalid dimensions.
        else:
            raise IndexError(f"mismatching shapes: y.shape = {y.shape} and y_pred.shape = {y_pred.shape}")
        # return -y.
        return -np.asarray(likelihoods, dtype=float).mean()

    def backward(self: NegativeLikelihood, y: Array1D, y_pred: Array1D, model: EasyNN.model.abc.Model = None) -> Any:
        """
        Updates self.derivatives (e.g. via backpropagation) for optimizers (e.g. gradient descent).
        If the last layer is the LogSoftMax activation, then a special algorithm is used.

        Parameters:
            y: Array1D
                The expected output of the model.
            y_pred: Array1D
                The predicted output from the model.
            model: Model
                The model being backpropagated into.

        Example:
            >>> model.backward_sample(x, y)           # For the user.
            >>> model.backward(loss.dy(y, model(x)))  # How it's implemented.
        """
        from EasyNN.model.activation.log_softmax import LogSoftMax
        use_y = isinstance(model.layers[-1], LogSoftMax)
        if model is None:
            model = self.model
        return model.backward(y_pred if use_y else self.dy(y, y_pred), y, use_y=use_y)

    def dy(self: NegativeLikelihood, y: Array1D, y_pred: Array1D) -> Array1D:
        # Check the inputs.
        y = np.array(y, copy=True)
        if y.ndim > 2:
            raise IndexError(
                "y must be either 0-D (the index of a single sample),\n"
                "    or 1-D (multiple indexes or an array of probabilities),\n"
                "    or 2-D (multiple arrays of probabilities),\n"
                "    but not {y.ndim}-D"
            )
        # Assumes y == 0 if it is not the answer and y == 1 if it is.
        # result = -1 if y matches the sample, else 0.
        result = np.zeros_like(y_pred)
        # y = [arrays of indexes] e.g. y = [2, 1].
        if y.ndim == 1 < y_pred.ndim:
            result[np.arange(len(y)), y] = -1
        # y = one index e.g. y = 2.
        elif y.ndim < y_pred.ndim:
            result[y] = -1
        # y = [arrays of 0 and 1] e.g. y = [0, 0, 1] or y = [[0, 0, 1], [0, 1, 0]].
        else:
            result[y != 0] = -1
        return result
