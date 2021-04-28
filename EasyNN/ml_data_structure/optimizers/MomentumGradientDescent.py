from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
from EasyNN.ml_data_structure.Momentum import Momentum


class MomentumGradientDescent(Optimizer):
    """Gradient descent optimizer with momentum."""

    learning_rate: float
    momentum: Momentum

    def __init__(self, learning_rate: float = 0.1, momentum_rate: float = 0.99) -> None:
        """
        Initialize momentum gradient descent by providing a learning rate and momentum rate.

        Parameters
        ----------
        learning_rate : float = 0.1
            The factor used on the derivatives when subtracting from the values.
        momentum_rate : float = 0.99
            The amount of the momentum preserved between iterations.
        """
        self.learning_rate = learning_rate
        self.momentum = Momentum(rate=momentum_rate)
        
    def update(self, model: "Model") -> None:
        """
        Subtracts learning_rate * momentum(derivatives) from the values.

        Parameters
        ----------
        model : Model
            The model to be optimized.
        model.values : np.ndarray
            The parameter values being optimized.
        model.derivatives : np.ndarray
            The parameter derivatives being optimized.
        """
        model.values -= self.learning_rate * self.momentum(model.derivatives)
