"""
TODO: Not complete.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from EasyNN.model.abc import Model
from EasyNN.typing import ArrayLikeND, ArrayND, Sample


class Loss(ABC):
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
    >>> accuracy = model.loss(batch.sample)
    >>> model.backward(model.loss.dy(batch.sample))
    """
    model: Model

    @abstractmethod
    def __call__(self: Loss, sample: Sample) -> float:
        """
        Calling loss() will provide the the loss value (i.e. accuracy in some contexts).

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
        >>> training_accuracy = loss(training.sample)
        >>> testing_accuracy = loss(testing.sample)
        """
        raise NotImplementedError

    def backward(self: Loss, sample: Sample) -> Any:
        """
        Updates self.derivatives (e.g. via backpropagation) for optimizers (e.g. gradient descent).
        Default implementation returns model.backward(loss.dy()).

        Parameters
        ----------
        sample: Sample = self.training.sample
            Uses the sample to compute the loss on.

        Example
        -------
        >>> # The following are equivalent:
        >>> model.backward()                       # For the user.
        >>> model.backward(model.loss.dy(sample))  # How it's implemented.
        >>> model.loss.backward(sample)            # For the optimizer.
        """
        return self.model.backward(self.dy(sample))

    @abstractmethod
    def dy(self: Loss, sample: Sample) -> ArrayND:
        """
        Computes the change in y expected, which is used to backpropagate into the model.
        Usually involves loss.y_pred and loss.y_true.

        Parameters
        ----------
        sample: Sample
            Uses the sample to compute the loss on.

        Returns
        -------
        dy: ArrayND
            The change in y that should be backpropagated into the model.
        """
        raise NotImplementedError

    def predict(self: Loss, sample: Sample) -> ArrayND:
        """
        Runs the model on the given sample. Usually just model(sample[0]).

        Parameters
        ----------
        sample: Sample
            Uses the sample to compute the loss on.

        Returns
        -------
        y_pred: ArrayND
            The result of the model on the given sample.
        """
        return self.model(sample[0])

    @property
    def derivatives(self: Loss) -> ArrayND:
        """Loss derivatives are a view into the underlying model derivatives."""
        return self.model.derivatives

    @derivatives.setter
    def derivatives(self: Loss, derivatives: ArrayLikeND) -> None:
        self.model.derivatives = derivatives

    @property
    def parameters(self: Loss) -> ArrayND:
        """Loss parameters are a view into the underlying model parameters."""
        return self.model.parameters

    @parameters.setter
    def parameters(self: Loss, parameters: ArrayLikeND) -> None:
        self.model.parameters = parameters
