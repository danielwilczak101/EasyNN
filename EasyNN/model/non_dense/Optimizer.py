from Tensor import TensorLike


class Optimizer:
    """Base Optimizer class for machine learning algorithms."""

    def update(self, iteration: int, values: TensorLike, derivatives: TensorLike):
        """Updates values using other parameters."""
        raise NotImplemented("Requires Optimizer.update(...) to be overriden.")
