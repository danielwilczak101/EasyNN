from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from EasyNN.batch.abc import Batch, Dataset
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
    >>> model = ...
    >>> model.loss = ...
    >>> x, y_true = ...
    >>> y_pred = model(x)
    >>> model.batch = MiniBatch((x, y_true))
    >>> accuracy = model.loss()
    >>> model.backward(model.loss.dy())
    """
    batch: Batch
    model: Model

    @abstractmethod
    def __call__(self: Loss, sample: Sample = None) -> float:
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
        >>> model = ...
        >>> model.loss = ...
        >>> training_accuracy = model.loss()
        >>> testing_accuracy = model.loss(model.loss.testing.sample)
        """
        raise NotImplementedError

    def backward(self: Loss, sample: Sample = None) -> Any:
        """
        Updates self.derivatives (e.g. via backpropagation) for optimizers (e.g. gradient descent).
        Default implementation returns model.backward(loss.dy()).

        Parameters
        ----------
        sample: Sample = self.training.sample
            Uses the sample to compute the loss on.

        Example
        -------
        >>> model = ...
        >>> model.loss = ...
        >>> # The following are equivalent:
        >>> model.backward()               # For the user.
        >>> model.backward(model.loss.dy)  # How it's implemented.
        >>> model.loss.backward()          # For the optimizer.
        """
        return self.model.backward(self.dy(sample))

    @abstractmethod
    def dy(self: Loss, sample: Sample = None) -> ArrayND:
        """
        Computes the change in y expected, which is used to backpropagate into the model.
        Usually involves loss.y_pred and loss.y_true.

        Parameters
        ----------
        sample: Sample = self.training.sample
            Uses the sample to compute the loss on.

        Returns
        -------
        dy: ArrayND
            The change in y that should be backpropagated into the model.
        """
        raise NotImplementedError

    def predict(self: Loss, sample: Sample = None) -> ArrayND:
        """
        Runs the model on the given sample. Usually just model(sample[0]).

        Parameters
        ----------
        sample: Sample = self.training.sample
            Uses the sample to compute the loss on.

        Returns
        -------
        y_pred: ArrayND
            The result of the model on the given sample.
        """
        if sample is None:
            sample = self.training.sample
        return self.model(sample[0])

    @property
    def training(self: Loss) -> Dataset:
        """Loss training dataset is a view into the underlying batch."""
        return self.batch.training

    @property
    def testing(self: Loss) -> Dataset:
        """Loss testing dataset is a view into the underlying batch."""
        return self.batch.testing

    @property
    def validation(self: Loss) -> Dataset:
        """Loss validation dataset is a view into the underlying batch."""
        return self.batch.validation

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
