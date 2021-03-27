from numpy import sqrt
from EasyNN.ml_data_structure.Tensor import TensorLike
from EasyNN.ml_data_structure.opimizers.Optimizer import Optimizer


class Adagrad(Optimizer):
    """Adagrad method."""
    learning_rate: float
    decay: float
    epsilon: float
    squares: TensorLike

    def __init__(self, *, learning_rate: float = 1.0, decay: float = 0.0, epsilon: float = 1e-7):
        """Initialize using learning rate, decay rate, and epsilon."""
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.squares = 0

    def update(self, iteration: int, items: Point):
        """Rescale the weight of each component in the derivatives using an accumulating cache."""

        self.squares += items.derivatives ** 2
        items.values -= (self.learning_rate / (1.0 + self.decay * iteration)) * items.derivatives / (np.sqrt(self.squares) + self.epsilon)
