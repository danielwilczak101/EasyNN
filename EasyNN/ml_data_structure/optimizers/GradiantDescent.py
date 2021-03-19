import Optimizer
from Point import Point


class GradiantDescent(Optimizer):
    """Basic Gradient Descent optimizer, using only a learning rate and derivatives."""
    
    def __init__(self, learning_rate: float = 0.1):
        """Requires a learning rate."""
        self.learning_rate = learning_rate

        
    def update(self, iteration: int, items: Point):
        """Subtract the derivatives times the learning rate from the values."""
        items.values -= self.learning_rate * items.derivatives
