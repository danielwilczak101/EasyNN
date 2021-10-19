"""
TODO: documentation.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from heapq import merge
from itertools import count, repeat
import numpy as np
from typing import Any, Callable, Generic, TypeVar, overload
from EasyNN._abc import AutoDocumentation
from EasyNN.batch.abc import Batch, Dataset
from EasyNN.batch.mini import MiniBatch
from EasyNN.loss.abc import Loss
from EasyNN.loss.mean_square_error import MeanSquareError
from EasyNN.optimizer.abc import Optimizer
from EasyNN.optimizer.adam import Adam
from EasyNN.classifier.classifier import Classifier, LabelType
from EasyNN.typing import Array1D, Array2D, Array3D, ArrayND, Callback, Command, Factory
from EasyNN.utilities.parameters import save,load

Labels = TypeVar("Labels")
ArrayIn = TypeVar("ArrayIn", bound=ArrayND)
ArrayOut = TypeVar("ArrayOut", bound=ArrayND)

@overload
def base_array(x: None) -> None:
    ...

@overload
def base_array(x: ArrayND) -> ArrayND:
    ...

def base_array(x):
    """Returns the base of the numpy array."""
    while x is not None is not x.base:
        x = x.base
    return x


class Model(AutoDocumentation, ABC, Generic[ArrayIn, ArrayOut]):
    """
    TODO: documentation.
    """
    _callbacks: dict[Command, list[Callback]]
    _command: Command = "off"
    _derivatives: Array1D
    _parameters: Array1D
    _x: ArrayIn
    _y: ArrayOut
    _training: Dataset[ArrayIn, ArrayOut]
    _testing: Dataset[ArrayIn, ArrayOut]
    _validation: Dataset[ArrayIn, ArrayOut]
    _classifier: Classifier[Labels]
    _optimizer: Optimizer
    _labels: Labels
    _default_batch: Factory[Batch] = MiniBatch
    _default_loss: Factory[Loss[ArrayIn, ArrayOut]] = MeanSquareError
    _default_optimizer: Factory[Optimizer] = Adam
    _default_classifier: Factory[Classifier] = Classifier
    stop_training: bool = False

    @property
    def command(self) -> Command:
        """Stores the current command being run."""
        return self._command

    @command.setter
    def command(self, command: Command) -> None:
        self._command = command
        if len(self.layers) == 1 and self.layers[0] is self:
            return
        for layer in self.layers:
            layer.command = command

    @property
    def layers(self) -> tuple[Model, ...]:
        """Returns the layers of the model."""
        # By default only one layer. Networks and DenseLayers should override this property.
        return (self,)

    @property
    def derivatives(self) -> Array1D:
        """Model derivatives are a 1D array used to store parameter derivatives during model.backward(dy)."""
        return self._derivatives

    @derivatives.setter
    def derivatives(self, derivatives: Array1D) -> None:
        if hasattr(self, "derivatives"):
            self._derivatives[...] = np.reshape(derivatives, -1)
        else:
            self._derivatives = np.asarray(derivatives, dtype=float).reshape(-1)

    @property
    def parameters(self) -> Array1D:
        """Model parameter values are a 1D array which can be modified to change the model."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Array1D) -> None:
        if hasattr(self, "parameters"):
            self._parameters[...] = np.reshape(parameters, -1)
        else:
            self._parameters = np.asarray(parameters, dtype=float).reshape(-1)

    @property
    def batch(self) -> Batch:
        """Used to generate samples from the dataset."""
        if not hasattr(self, "_batch"):
            self.batch = self._default_batch()
        return self._batch

    @batch.setter
    def batch(self, batch: Batch) -> None:
        self._batch = batch

    @property
    def classifier(self) -> Classifier[T]:
        """Used to classify the outputs of the model."""
        if not hasattr(self, "_classifier"):
            self.classifier = self.loss._default_classifier()
        return self._classifier

    @classifier.setter
    def classifier(self, classifier: Classifier[T]) -> None:
        self._classifier = classifier

    @property
    def loss(self) -> Loss[ArrayIn, ArrayOut]:
        """Reference to the loss stored in the optimizer."""
        if not hasattr(self, "_loss"):
            self.loss = self._default_loss()
        return self._loss

    @loss.setter
    def loss(self, loss: Loss[ArrayIn, ArrayOut]) -> None:
        self._loss = copy(loss)
        self.loss.model = self

    @property
    def labels(self) -> dict[int, Labels]:
        return self.training.labels

    @labels.setter
    def labels(self, labels) -> None:
        """Sets the labels when asked for them."""
        self.training.labels = labels

    @property
    def optimizer(self) -> Optimizer:
        """Stores the optimizer being used for training."""
        if not hasattr(self, "_optimizer"):
            self.optimizer = self._default_optimizer()
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        optimizer.add(self)
        self._optimizer = optimizer

    @property
    def iteration(self) -> int:
        """Returns the current iteration from training."""
        return self.training.iteration

    @property
    def epoch(self) -> int:
        """Returns the current epoch from training."""
        return self.training.epoch

    @property
    def training(self) -> Dataset:
        """
        The training dataset the main dataset used during training.
        Use model.training.data = (x, y) to set the training data.
        """
        if not hasattr(self, "_training"):
            self._training = Dataset()
        return self._training

    @property
    def testing(self) -> Dataset:
        """
        The testing dataset is used to produce unbiased evaluation of the model.
        Use model.tsting.data = (x, y) to set the testing data.
        """
        if not hasattr(self, "_testing"):
            self._testing = Dataset()
        return self._testing

    @property
    def validation(self) -> Dataset:
        """
        The validation dataset is used as a pseudo testing dataset during training.
        Use model.validation.data = (x, y) to set the validation data.
        """
        if not hasattr(self, "_validation"):
            self._validation = Dataset()
        return self._validation

    @property
    def x(self) -> ArrayIn:
        """
        Stores the x values from the previous call for future use,
        such as for backpropagation.
        """
        return self._x

    @x.setter
    def x(self, x: ArrayIn) -> None:
        self._x = np.asarray(x, dtype=float)

    @property
    def y(self) -> ArrayOut:
        """
        Stores the y values from the previous call for future use,
        such as for backpropagation.
        """
        return self._y

    @y.setter
    def y(self, y: ArrayOut) -> None:
        self._y = np.asarray(y, dtype=float)

    def get_arrays(self) -> dict[str, ArrayND]:
        """Returns the arrays stored in the model."""
        return dict(parameters=self.parameters)

    def set_arrays(self, *, parameters: ArrayND = None) -> None:
        """Sets the arrays stored in the model."""
        if parameters is not None:
            self.parameters = parameters

    def prepare_datasets(self) -> None:
        """Prepares the datasets before optimizing."""
        if not hasattr(self.training, "data"):
            raise ValueError("requires model.training.data to be set to train the model")
        elif len(self.training) < 10:
            raise ValueError("requires len(model.training.data) >= 10 to train the model")
        # Shuffle the training dataset.
        self.training.data = self.training[np.random.permutation(len(self.training))]
        # Steal 15% of the testing data from the training data if necessary.
        if not hasattr(self.testing, "data"):
            self.testing.data = self.training[:int(len(self.training) * 0.15)]
            self.training.data = self.training[int(len(self.training) * 0.15):]
        # Steal 15% of the validation data from the training data if necessary.
        if not hasattr(self.validation, "data"):
            self.validation.data = self.training[:int(len(self.training) * 0.15)]
            self.training.data = self.training[int(len(self.training) * 0.15):]
        # Apply the batches to each dataset.
        self.training.batch = self.batch
        self.testing.batch = MiniBatch(len(self.testing))
        self.validation.batch = MiniBatch(256)
        for layer in self.layers:
            layer._training = self.training
            layer._testing = self.testing
            layer._validation = self.validation

    def fit(self, x: ArrayIn, y: ArrayOut) -> None:
        """
        Fit the model to the provided training data.

        Equivalent to setting the training.data, and then using model.train().
        """
        self.training.data = (x, y)
        self.train()

    def train(self) -> None:
        """Train the model."""
        self.optimizer.train(self)

    def callback(self, command: Command) -> Callable[[Callback], Callback]:
        """model.callback(...) returns a decorator for saving callbacks."""
        def get_callback(cb: Callback) -> Callback:
            # Save the new callback to the given command.
            self.callbacks[command].append(cb)
            # Return the callback for decorator usage.
            return cb
        # Return the decorator.
        return get_callback

    def on_optimization_start(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_optimization_start')."""
        self.callbacks["on_optimization_start"].append(cb)
        return cb

    def on_optimization_end(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_optimization_end')."""
        self.callbacks["on_optimization_end"].append(cb)
        return cb

    def on_training_start(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_training_start')."""
        self.callbacks["on_training_start"].append(cb)
        return cb

    def on_training_end(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_training_end')."""
        self.callbacks["on_training_end"].append(cb)
        return cb

    def on_testing_start(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_testing_start')."""
        self.callbacks["on_testing_start"].append(cb)
        return cb

    def on_testing_end(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_testing_end')."""
        self.callbacks["on_testing_end"].append(cb)
        return cb

    def on_validation_start(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_validation_start')."""
        self.callbacks["on_validation_start"].append(cb)
        return cb

    def on_validation_end(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_validation_end')."""
        self.callbacks["on_validation_end"].append(cb)
        return cb

    def on_epoch_end(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_epoch_end')."""
        self.callbacks["on_epoch_end"].append(cb)
        return cb

    def on_epoch_start(self, cb: Callback) -> Callback:
        """Shortcut for model.callback('on_epoch_start')."""
        self.callbacks["on_epoch_start"].append(cb)
        return cb

    def training_validation_commands(self) -> Iterator[Command]:
        """Generates the training/validation commands in an even manner."""
        # Set commands = (step, "start"), (step, "end"), (2 * step, "start"), (2 * step, "end"), ...
        # The step is used to control how frequently each command occurs.
        training_step = len(self.validation) * self.training._batch_size
        training_commands = (
            (i, command)
            for i in count(start=training_step, step=training_step)
            for command in ("on_training_start", "on_training_end")
        )
        validation_step = len(self.training) * self.validation._batch_size
        validation_commands = (
            (i, command)
            for i in count(start=validation_step, step=validation_step)
            for command in ("on_validation_start", "on_validation_end")
        )
        # Sort the commands by the steps.
        for step, command in merge(training_commands, validation_commands, key=lambda step_command: step_command[0]):
            yield command

    def _optimizer_commands(self) -> Iterator[Command]:
        """
        Generates all of the commands for the optimizer.

        Follows the following sequence of events:
            optimization_start -> training/validation -> optimization_end -> testing_start -> testing_end
        """
        # Start optimizing.
        yield "on_optimization_start"
        # Track the epoch to know when it changes.
        epoch = self.training.epoch
        yield "on_epoch_start"
        # Loop through the training/validation commands until training stops.
        for command in self.training_validation_commands():
            yield command
            if self.stop_training:
                break
            # Switch between epochs whenever the training epoch changes.
            if -1 < epoch < self.training.epoch:
                yield "on_epoch_end"
                yield "on_epoch_start"
            epoch = self.training.epoch
        # Trigger on_{training/validation}_end and on_epoch_end when training stops.
        if command.endswith("_start"):
            yield command.replace("_start", "_end")
        yield "on_epoch_end"
        # Stop optimizing.
        yield "on_optimization_end"
        # Test the results.
        yield "on_testing_start"
        yield "on_testing_end"

    def optimizer_commands(self) -> Iterator[Command]:
        """Generates all of the commands and runs their callbacks for the optimizer."""
        # Prepare the datasets before optimizing.
        self.prepare_datasets()
        # Run all of the commands.
        for self.command in self._optimizer_commands():
            # Run all of the callbacks.
            for callback in self.callbacks[self.command]:
                callback(self)
            yield self.command
        self.command = "off"

    def accuracy(self, x: ArrayIn=None, y: ArrayOut=None) -> float:
        """Returns the classification accuracy of the data."""
        # If the user doesnt specify the dataset. We will try to give it them using the training data
        try:
            return self.classifier.accuracy(self(x), y)
        except:
            return self.classifier.accuracy(self(self.training.data[0]), self.training.data[1])
        else:
            print("Could not load training data to calculate accuracy data.")
            raise
            
    def classify(self, x: ArrayIn) -> T:
        """Returns the classification of the input data."""
        return self.classifier.classify(self(x), self.labels)

    def sample_derivatives(self, x: ArrayIn, y: ArrayOut) -> None:
        """Shortcut for `model.loss.backward(y, model(x), model)`."""
        self.loss.backward(y, self(x), self)

    def __call__(self, x: ArrayIn) -> ArrayOut:
        """
        Implements y = model(x) for feed-forward propagation.
        Parses the input as a numpy array before using the forward implementation method.
        """
        self.x = x
        if not hasattr(self, "parameters"):
            self.__setup__()
        self.y = self.__forward__(self.x)
        if not hasattr(self, "parameters"):
            self.__setup__()
        return self.y

    def backward(self, dy: ArrayOut, y: ArrayOut = None, use_y: bool = False) -> ArrayIn:
        """
        Implements dx = model.backward(dy) for backpropagation.
        Parses the input as a numpy array before using the backward implementation method.
        """
        return self.__backward__(
            dy=np.asarray(dy),
            y=(None if y is None else np.asarray(y)),
            use_y=use_y,
        )

    @abstractmethod
    def __backward__(self, dy: ArrayOut, y: ArrayOut = None, use_y: bool = False) -> ArrayIn:
        """Implements the backpropagation after the input has been parsed."""
        raise NotImplementedError

    @abstractmethod
    def __forward__(self, x: ArrayIn) -> ArrayOut:
        """Implements the feed-forward propagation after the input has been parsed."""
        raise NotImplementedError

    @abstractmethod
    def __setup__(self) -> None:
        """
        Implements the setup procedure for the values, derivatives, shapes in, and shapes
        out.
        """
        raise NotImplementedError


class Model_1D(Model[Array1D, Array1D], ABC):
    """Base model for handling 1D inputs and outputs."""


class Model_2D(Model[Array2D, Array2D], ABC):
    """Base model for handling 2D inputs and outputs."""


class Model_3D(Model[Array3D, Array3D], ABC):
    """Base model for handling 3D inputs and outputs."""
