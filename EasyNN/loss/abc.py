"""
TODO: Not complete.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from EasyNN._abc import AutoDocumentation
# TODO: from EasyNN.classifier.abc import Classifier
from EasyNN.typing import ArrayND, Factory
import EasyNN.model.abc


class Loss(AutoDocumentation, ABC):
    """
    Abstract Base Class for the Loss, which measures the performance of a model,
    provided batches of data. The Loss is also used to compute derivatives,
    which are backpropagated into the model.

    This class is used between the Model and the Optimizer.

    Example
    -------
    >>> from EasyNN.batch.mini import MiniBatch
    >>> model = ...
    >>> model.loss = ...
    >>> batch = MiniBatch(...)
    >>> x, y_true = batch.sample
    >>> y_pred = model(x)
    >>> accuracy = model.loss(*batch.sample)
    >>> model.backward(model.loss.dy(*batch.sample))
    """
    model: EasyNN.model.abc.Model
    _default_classifier: Factory[Classifier]  # TODO: Make default classifier.

    @abstractmethod
    def __call__(self: Loss, x: Array1D, y: Array1D) -> float:
        """
        Calling loss(x, y) will provide the the loss value.

        Parameters
        ----------
        sample: Sample = self.training.sample
            Uses the sample to compute the loss on.

        Returns
        -------
        loss_value: float
            A measurement of how good the model is. This value should be minimized.

        Example
        -------
        >>> training_accuracy = loss(*training.sample)
        >>> testing_accuracy = loss(*testing.sample)
        """
        raise NotImplementedError

    def backward(self: Loss, y: ArrayND, y_pred: ArrayND, model: EasyNN.model.abc.Model[ArrayND, ArrayND] = None) -> Any:
        """
        Updates self.derivatives (e.g. via backpropagation) for optimizers (e.g. gradient descent).
        Default implementation returns model.backward(loss.dy(y, model(x))).

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
        if model is None:
            model = self.model
        return model.backward(self.dy(y, y_pred))

    @abstractmethod
    def dy(self: Loss, y: ArrayND, y_pred: ArrayND) -> ArrayND:
        """
        Computes the change in y expected, which is used to backpropagate into the model.
        Usually involves loss.y_pred and loss.y_true.

        Parameters:
            y, y_pred: Array1D
                The expected and predicted y values.

        Returns:
            dy: ArrayND
                The change in y that should be backpropagated into the model.
        """
        raise NotImplementedError
