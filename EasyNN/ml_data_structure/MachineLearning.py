from Tensor import TensorLike
from Optimizer import Optimizer


class MachineLearning:
    """
    Stores value/derivative data that is trained using a machine
    learning algorithm from the optimizer object. 
    """

    iteration: int
    """Number of times self.optimize() was called."""

    items: TensorLike
    """2D array of [values, derivatives], holding all parameter references."""

    optimizer: Optimizer
    """Algorithm for optimizing values."""

    def __init__(self, items: TensorLike, optimizer: Optimizer):
        """Initialize a ML object with values, derivatives, and an optimizer."""
        self.iteration = 0
        self.items = items
        self.optimizer = optimizer

    def optimize(self):
        """Optimize the values given the values and derivatives, using the optimizer."""
        self.iteration += 1
        self.optimizer.update(self.iteration, self.values, self.derivatives)

    @property
    def values(self) -> TensorLike:
        """The weights and biases of the model."""
        return self.items[0]

    @values.setter
    def values(self, input_values: TensorLike):
        """Assign values while preserving references."""
        self.items[0] = input_values

    @property
    def derivatives(self) -> TensorLike:
        """The derivatives of the weights and biases of the model."""
        return self.items[1]

    @derivatives.setter
    def derivatives(self, input_derivatives: TensorLike):
        """Assign derivatives while preserving references."""
        self.items[1] = input_derivatives
