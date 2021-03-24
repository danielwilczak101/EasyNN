from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.Point import Point


class GradientDescent(Optimizer):
    """Basic Gradient Descent optimizer, using only a learning rate and derivatives."""

    learning_rate: float
    """Rate at which the values change based on the derivatives."""

    def __init__(self, learning_rate: float = 0.1):
        """Requires a learning rate."""
        self.learning_rate = learning_rate

        
    def update(self, iteration: int, items: Point):
        """Subtract the derivatives times the learning rate from the values."""
        items.values -= self.learning_rate * items.derivatives
