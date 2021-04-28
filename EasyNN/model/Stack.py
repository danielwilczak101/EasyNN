import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Iterable, Tuple, List
from EasyNN.model.Model import Model
from EasyNN.model.Weights import Weights
from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.optimizers.GradientDescent import GradientDescent
from EasyNN.Batch import Batch, MiniBatch
from EasyNN.loss.Loss import Loss
from EasyNN.loss.MeanSquareError import MeanSquareError


class Stack(Model):
    """Stack several models into one."""
    models: List[Model]


    def __init__(self, model: Union[Model, Iterable[Model]], *models: Tuple[Model, ...]) -> None:
        """
        Initialize the stack by taking in several models and recreating their parameters.

        Parameters
        ----------
        model : Union[Model, Iterable[Model]]
            Take in one model or an iterable of models.
        models : Tuple[Model, ...]
            Allow models to be passed in as args instead.

        Notes
        -----

        Examples
        --------
        >>> model = Stack([model1, model2, model3])
        >>> model = Stack(model4, model5)
        """
        if isinstance(model, Model):
            self.models = [model] + list(models)
        else:
            self.models = list(model) + list(models)
        self.parameters = np.hstack([model.parameters for model in self.models])


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
        for model in self.models:
            values = model(values)
        return values


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
        for model in reversed(self.models):
            derivatives = model.backpropagate(derivatives)
        return derivatives


    def train(
            self,
            inputs: ArrayLike,
            outputs: ArrayLike,
            epochs: int = 1000,
            optimizer: Optimizer = GradientDescent(),
            batches: Batch = MiniBatch(32),
            loss: Loss = MeanSquareError(),
            l1_rate: float = 0.01,
            l2_rate: float = 0.01,
        ) -> None:
        """
        Trains the model using inputs and outputs.

        Parameters
        ----------
        inputs : ArrayLike
            Values plugged into the model.
        outputs : ArrayLike
            Expected outputs from the model.
        epochs : int = 1000
            Number of times the entire dataset is passed through.
        optimizer : Optimizer
            The optimizer used to update the parameters.
        batches : Batch = MiniBatch(32)
            How batches are extracted from the dataset.
        loss : Loss = MeanSquareError()
            The cost function used for computing loss error and
            derivatives to be backpropagated from.
        l1_rate : float = 0.01
            The coefficient for L1 regularization.
        l2_rate : float = 0.01
            The coefficient for L2 regularization.
        """
        inputs = np.array(inputs, copy=False)
        outputs = np.array(outputs, copy=False)

        for i in range(epochs):
            for indexes in batches(len(inputs)):
                batch = inputs.take(indexes, axis=0)
                expectation = outputs.take(indexes, axis=0)
                prediction = self(batch)
                print('cost: ', loss.cost(batch, expectation, prediction))
                self.backpropagate(loss.gradient(batch, expectation, prediction))
                for model in self.individual_models:
                    if isinstance(model, Weights):
                        model.derivatives += l1_rate * np.sign(model.values)
                        model.derivatives += l2_rate * model.values
                optimizer.update(self)

    @property
    def individual_models(self) -> Iterable[Model]:
        """Iterator over individual models i.e. 'unstacking'."""
        for model in self.models:
            if isinstance(model, Stack):
                yield from model.individual_models
            else:
                yield model


    @property
    def parameters(self) -> np.ndarray:
        """Property for parameters stored in the stack, updates all models when changed."""
        return self._parameters


    @parameters.setter
    def parameters(self, new_parameters: np.ndarray) -> None:
        """Update all of the hidden model parameters to the new one."""
        self._parameters = new_parameters
        cursor = 0
        for model in self.models:
            model.parameters = new_parameters[:, cursor:cursor+len(model.values)]
            cursor += len(model.values)
