from numpy import sqrt
from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer
import numpy as np

class Adagrad(Optimizer):
    """Adagrad method."""
    learning_rate: float
    epsilon: float
    squares: np.ndarray


    def __init__(self, learning_rate: float = 0.1, epsilon: float = 1e-7) -> None:
        """
        Initialize Adagrad rates.

        Parameters
        ----------
        learning_rate : float = 0.1
            The factor used on the derivatives when subtracting from the values.
        epsilon : float = 1e-7
            Used to avoid division by 0 in the denominator of derivatives / sqrt(squares).
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon


    def update(self, model: "Model") -> None:
        """
        Subtracts learning_rate * derivatives from the values.

        Parameters
        ----------
        model : Model
            The model to be optimized.
        model.values : np.ndarray
            The parameter values being optimized.
        model.derivatives : np.ndarray
            The parameter derivatives being optimized.
        """
        if hasattr(self, 'squares'):
            self.squares += model.derivatives ** 2
        else:
            self.squares = model.derivatives ** 2
        model.values -= self.learning_rate * model.derivatives / (np.sqrt(self.squares) + self.epsilon)
