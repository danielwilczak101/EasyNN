from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.Point import Point
from EasyNN.ml_data_structure.Momentum import Momentum


class MomentumGradientDescent(Optimizer):
    """Gradient descent optimizer with momentum."""

    learning_rate: float
    momentum: Momentum

    def __init__(self, learning_rate: float = 0.1, momentum_rate: float = 0.99):
        """Requires a learning rate."""
        self.learning_rate = learning_rate
        self.momentum = Momentum(rate=momentum_rate)
        
    def update(self, iteration: int, items: Point):
        """Subtract the derivatives times the learning rate from the values."""
        items.values -= self.learning_rate * self.momentum(items.derivatives)
