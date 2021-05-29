from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import ArrayLike
import numpy as np
from EasyNN import Handler, handler
from EasyNN.IO import IO
from EasyNN.Optimizer import Optimizer, GradientDescent
from EasyNN.Cost import Cost, MeanSquareError
from EasyNN.Batch import Batch, MiniBatch


@handler
@dataclass(eq=False)
class Model:
    """
    Abstract Model requires the following to be implemented:
    - Model.ff() and/or Model.ff_parallel().
    - Model.bp() and/or Model.bp_parallel().
    - Model.is_parallel_inputs.getter().
    """

    #============#
    # Attributes #
    #============#

    # use the global handler by default
    handler: Handler = field(default=handler, init=False, repr=False)
    # IO are initialized as IO objects
    inputs: IO = field(default_factory=IO, init=False, repr=False)
    outputs: IO = field(default_factory=IO, init=False, repr=False)
	# the optimizer is gradient descent by default
    optimizer: Optimizer = GradientDescent
    # the cost is the mean square error by default
    cost: Cost = MeanSquareError
    # the batch is the mini batch by default
    batch: Batch = MiniBatch
    # values and derivatives are only included for type-hints
    values: np.ndarray = field(init=False, repr=False)
    derivatives: np.ndarray = field(init=False, repr=False)
    # parameters are initialized as empty values and derivatives
    parameters: np.ndarray = field(default_factory=lambda: np.empty((2, 0)))

    def pre_setup(self: Model) -> None:
        """
        Setup the model after updating self.inputs.values and before
        the call is finished for models not fully setup after __init__.
        """
        pass

    def post_setup(self: Model) -> None:
        """
        Setup the model after updating self.inputs.values and after
        the call is finished for models not fully setup after __init__.
        """
        pass

    #==================================#
    # Feed-forward propagation methods #
    #==================================#

    def __call__(self: Model, values: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Implements Model(values) -> outputs.values.
        - Collects inputs.values.
        - Runs Model.setup() for runtime model setup.
        - Runs either Model.ff or Model.ff_parallel, depending on the input.
        - Collects outputs.values.
        - Returns outputs.values.

        Parameters
        ----------
        values : Optional[ArrayLike] = None
            The input values taken.
            If None, uses the default inputs provided from e.g. Model_1.inputs = Model_2.outputs;

        Returns
        -------
        self.outputs.values : np.ndarray
            The output values after performing feedforward propagation.
        """
        # store input values
        self.inputs.values = values
        # setup the model before the call is carried out
        self.pre_setup()
        # compute output values
        if self.is_parallel_inputs:
            self.outputs.values = self.ff_parallel(self.inputs.values)
        else:
            self.outputs.values = self.ff(self.inputs.values)
        # setup the model after the call is carried out
        self.post_setup()
        return self.outputs.values

    def ff(self: Model, values: np.ndarray) -> ArrayLike:
        """Implements Model(values) -> outputs.values when only one input is given."""
        if self.ff_parallel.__func__ is Model.ff_parallel:
            raise NotImplemented("feedforward propagation needs to be implemented for Model.ff() or Model.ff_parallel()")
        # by default, use the parallel method
        return self.ff_parallel(np.expand_axis(values, 0))[0]

    def ff_parallel(self: Model, values: np.ndarray) -> ArrayLike:
        """Implements Model([values, ...]) -> outputs.values when multiple inputs are given."""
        if self.ff.__func__ is Model.ff:
            raise NotImplemented("feedforward propagation needs to be implemented for Model.ff() or Model.ff_parallel()")
        # by default, use the single method
        return np.apply_along_axis(self.ff, 0, values)

    #==========================#
    # Back propagation methods #
    #==========================#

    def backpropagate(self: Model, derivatives: Optional[ArrayLike] = None) -> np.ndarray:
        # store output derivatives
        self.outputs.derivatives = derivatives
        # compute input derivatives and update self.derivatives
        if self.is_parallel_inputs:
            self.inputs.derivatives = self.bp_parallel(self.outputs.derivatives)
        else:
            self.inputs.derivatives = self.bp(self.outputs.derivatives)
        return self.inputs.derivatives

    def bp(self: Model, derivatives: np.ndarray) -> ArrayLike:
        if self.bp_parallel.__func__ is Model.bp_parallel:
            raise NotImplemented("back propagation needs to be implemented for Model.bp() or Model.bp_parallel()")
        # by default, use the parallel method
        return self.bp_parallel(np.extend_axis(derivatives, 0))[0]

    def bp_parallel(self: Model, derivatives: np.ndarray) -> ArrayLike:
        if self.bp.__func__ is Model.bp:
            raise NotImplemented("back propagation needs to be implemented for Model.bp() or Model.bp_parallel()")
        # by default, use the single method
        # and sum up the derivatives
        # see Model.derivatives.setter
        self.derivatives = 0
        return np.apply_along_axis(self.bp, 0, derivatives)


    def train(
        self: Model,
        training_inputs: ArrayLike,
        training_outputs: ArrayLike,
        testing_inputs: Optional[ArrayLike] = None,
        testing_outputs: Optional[ArrayLike] = None,
        validation_inputs: Optional[ArrayLike] = None,
        validation_outputs: Optional[ArrayLike] = None,
        testing_percent: float = 0.15,
        validation_percent: float = 0.15,
    ) -> None:
        # verify both inputs and outputs are provided
        if testing_inputs is None ^ testing_outputs is None:
            raise ValueError("testing inputs and testing outputs must both be defined or both be undefined")
        if validation_inputs is None ^ validation_outputs is None:
            raise ValueError("validation inputs and validation outputs must both be defined or both be undefined")
        # data needs to be pulled out of the training data
        if testing_inputs is None or validation_inputs is None:
            # cast training data to numpy arrays
            training_inputs = np.array(training_inputs, copy=False)
            training_outputs = np.array(training_outputs, copy=False)
            # default testing/validation data requires training data to be shuffled before pulling out values
            indexes = np.random.permutation(len(training_inputs))
            training_inputs = training_inputs[indexes]
            training_outputs = training_outputs[indexes]
        	# extract default testing data
        	if testing_inputs is None:
                # ensure validation percentage is not affected by testing percentage
        	    validation_amount = len(training_inputs) * validation_percent
                # extract last section of training data for testing data
           		testing_inputs = training_inputs[-int(len(training_inputs)*testing_percent):]
            	testing_outputs = training_outputs[-int(len(training_inputs)*testing_percent):]
                # remove last section from training data
            	training_inputs = training_inputs[:-int(len(training_inputs)*testing_percent)]
            	training_outputs = training_outputs[:-int(len(training_inputs)*testing_percent)]
                # ensure validation percentage is not affected by testing percentage
            	validation_percent = validation_amount / len(training_inputs)
        	# extract default validation data
        	if validation_inputs is None:
                # extract last section of training data for validation data
            	validation_inputs = training_inputs[-int(len(training_inputs)*validation_percent):]
            	validation_outputs = training_outputs[-int(len(training_inputs)*validation_percent):]
                # remove last section from training data
            	training_inputs = training_inputs[:-int(len(training_inputs)*validation_percent)]
            	training_outputs = training_outputs[:-int(len(training_inputs)*validation_percent)]
        # verify there's some training data
        if len(training_inputs) == 0:
            raise ValueError("no training data, not including the testing and validation data.")
		# store batch data
        self.batch.training.data = (training_inputs, training_outputs)
        self.batch.testing.data = (testing_inputs, testing_outputs)
        self.batch.validation.data = (validation_inputs, validation_outputs)
		# run optimizer
        self.optimizer.run()

    #============#
    # Properties #
    #============#

    @property
    def values(self: Model) -> np.ndarray:
        return self.parameters[0]

    @values.setter
    def values(self: Model, values: ArrayLike) -> None:
        # flatten the values before setting them
        self.parameters[0] = np.reshape(values, -1)

    @property
    def derivatives(self: Model) -> np.ndarray:
        return self.parameters[1]

    @derivatives.setter
    def derivatives(self: Model, derivatives: ArrayLike) -> None:
        # if parallel derivatives using default Model.bp_parallel,
        # the derivatives are flattened and summed
        if self.is_parallel_inputs and self.bp_parallel.__func__ is Model.bp_parallel:
            self.parameters[1] += np.reshape(derivatives, -1)
        # otherwise just use the flattened derivatives
        else:
            self.parameters[1] = np.reshape(derivatives, -1)

    @property
    def is_parallel_inputs(self: Model) -> bool:
        # use the non-parallel ff and bp methods by default.
        return False

    @property
    def inputs(self: Model) -> IO:
        return self._inputs

    @inputs.setter
    def inputs(self: Model, inputs: IO) -> None:
        # set the source of the inputs
        self.inputs.values_source = inputs.values_source
        inputs.setup(self.inputs)

    @property
    def optimizer(self: Model) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self: Model, optimizer: Union[str, type, Optimizer]) -> None:
        # use self.handler on strings
        if isinstance(optimizer, str):
            optimizer = self.handler[optimizer]
        # instantiate classes with no args
        if isinstance(optimizer, type):
            optimizer = optimizer()
        # use self.cost if the optimizer doesn't have a cost
        if not hasattr(optimizer, "cost") and hasattr(self, "cost"):
            optimizer.cost = self.cost
        self._optimizer = optimizer

    @property
    def cost(self: Model) -> Cost:
        return self.optimizer.cost

    @cost.setter
    def cost(self: Model, cost: Union[str, type, Cost]) -> None:
        # use self.handler on strings
        if isinstance(cost, str):
            cost = self.handler[cost]
        # instantiate classes with no args
        if isinstance(cost, type):
            cost = cost()
        # use self.batch if the cost doesn't have a batch
        if not hasattr(cost, "batch") and hasattr(self, "batch"):
            cost.batch = self.batch
        self.optimizer.cost = cost

    @property
    def batch(self: Model) -> Batch:
        return self.cost.batch

    @batch.setter
    def batch(self: Model, batch: Union[str, type, Batch]) -> None:
        # use self.handler on strings
        if isinstance(batch, str):
            batch = self.handler[batch]
        # instantiate classes with no args
        if isinstance(batch, type):
            batch = batch()
        self.cost.batch = batch
