from EasyNN.ml_data_structure.optimizers.Optimizer import Optimizer


class GradientDescent(Optimizer):
    """Basic Gradient Descent optimizer, using only a learning rate and derivatives."""

    learning_rate: float
    """Rate at which the values change based on the derivatives."""

    def __init__(self, learning_rate: float = 0.1) -> None:
        """
        Initialize gradient descent by providing a learning rate.

        Parameters
        ----------
        learning_rate : float = 0.1
            The factor used on the derivatives when subtracting from the values.
        """
        self.learning_rate = learning_rate

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
        model.values -= self.learning_rate * model.derivatives
