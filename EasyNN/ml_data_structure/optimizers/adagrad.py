from Optimizer import Optimizer


class StochasticAdagrad(Optimizer):
    """Adagrad method with stochastic learning rate."""

    learning_rate: float
    decay: float
    epsilon: float

    def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, epsilon: float = 1e-7):
        """Initialize using learning rate, decay rate, and epsilon."""
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None

    def update(self, iteration: int, items: Point):
        """Rescale the weight of each component in the derivatives using an accumulating cache."""
        
        # Initialize the cache if necessary
        if self.cache is None:
            self.cache = np.zeros_like(items.values)

        self.cache += items.derivatives ** 2
        items.values -= (self.learning_rate / (1.0 + self.decay * iteration)) * items.derivatives / (np.sqrt(self.cache) + self.epsilon)

